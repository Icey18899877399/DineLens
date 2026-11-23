import argparse
import json
import logging
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qs, urlsplit

from aspect_detection.evidence import load_evidence_extractor
from aspect_detection.pipeline import load_predictor
from aspect_detection.visualization_data import load_visualization_data

from .catalog import ReviewCatalog

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = Path(__file__).with_name("public")
RESULTS = ROOT / "runs" / "autodl-20261123" / "runs"
MAX_REVIEW_LENGTH = 4_000


class ReviewApp:
    def __init__(self):
        self.catalog = ReviewCatalog.from_files(
            ROOT / "dataset" / "test.csv", RESULTS / "joint_bert" / "test_predictions.csv"
        )
        self.overview = self.catalog.overview()
        self.metrics = load_visualization_data(RESULTS)
        self._predictor = None
        self._extractor = None
        self._lock = Lock()

    def _with_evidence(self, data):
        with self._lock:
            if self._extractor is None:
                self._extractor = load_evidence_extractor(RESULTS / "aspect_baseline")
        return {
            **data,
            "aspects": self._extractor.annotate_predictions(data["text"], data["aspects"]),
        }

    def detail(self, review_id):
        return self._with_evidence(self.catalog.detail(review_id))

    def analyze(self, review):
        if not isinstance(review, str) or not review.strip():
            raise ValueError("请先输入一条评论")
        if len(review) > MAX_REVIEW_LENGTH:
            raise ValueError(f"评论请控制在 {MAX_REVIEW_LENGTH:,} 字以内")
        with self._lock:
            if self._predictor is None:
                self._predictor = load_predictor(RESULTS / "joint_bert", device="cpu")
            rows = self._predictor.predict(review)
        return self._with_evidence({"text": review, "aspects": rows})


def serve(app, port):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(PUBLIC), **kwargs)

        def _json(self, value, status=200):
            content = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            url = urlsplit(self.path)
            if not url.path.startswith("/api/"):
                return super().do_GET()
            query = parse_qs(url.query)
            try:
                if url.path == "/api/overview":
                    return self._json(app.overview)
                if url.path == "/api/metrics":
                    return self._json(app.metrics)
                if url.path == "/api/reviews":
                    polarity = query.get("sentiment", [None])[0]
                    return self._json(app.catalog.browse(
                        query.get("aspect", [None])[0],
                        int(polarity) if polarity is not None else None,
                        int(query.get("offset", ["0"])[0]),
                    ))
                if url.path == "/api/review":
                    return self._json(app.detail(query.get("id", [""])[0]))
                return self._json({"message": "找不到该页面"}, 404)
            except KeyError:
                return self._json({"message": "这条评论未找到"}, 404)
            except ValueError:
                return self._json({"message": "评论筛选参数无效"}, 400)
            except Exception:
                logging.exception("Unable to load review")
                return self._json({"message": "暂时无法读取评论，请重试"}, 503)

        def do_POST(self):
            if urlsplit(self.path).path != "/api/analyze":
                return self._json({"message": "找不到该页面"}, 404)
            if self.headers.get_content_type() != "application/json":
                return self._json({"message": "请求格式无效"}, 415)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 100_000:
                    return self._json({"message": "请求内容为空或过长"}, 400)
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("请求格式无效")
                return self._json(app.analyze(request.get("review")))
            except (UnicodeError, json.JSONDecodeError):
                return self._json({"message": "请求格式无效"}, 400)
            except ValueError as exc:
                return self._json({"message": str(exc)}, 400)
            except Exception:
                logging.exception("Unable to analyze review")
                return self._json({"message": "本地模型暂时不可用，请稍后重试"}, 503)

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Local: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动餐前点评本地网页")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(ReviewApp(), args.port)

