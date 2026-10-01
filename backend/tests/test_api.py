"""The HTTP API walks the same checkpoints as the service."""

from fastapi.testclient import TestClient

from eval_factory.api.app import create_app
from eval_factory.config import Settings


def _client(tmp_path) -> TestClient:
    app = create_app(Settings(data_dir=tmp_path / "data"))
    return TestClient(app)


def test_health_and_ontology(tmp_path):
    client = _client(tmp_path)
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    ontology = client.get("/api/ontology")
    assert ontology.status_code == 200
    types = {item["system_type"] for item in ontology.json()}
    assert types == {"rag", "chatbot", "agentic"}


def test_api_flow_publishes_a_report_and_charts(tmp_path):
    client = _client(tmp_path)
    created = client.post("/api/runs", json={"name": "API RAG", "system_type": "rag"})
    assert created.status_code == 201
    run_id = created.json()["id"]

    rejected = client.post(
        f"/api/runs/{run_id}/artifacts",
        files={"file": ("notes.exe", b"nope", "application/octet-stream")},
    )
    assert rejected.status_code == 400

    uploaded = client.post(
        f"/api/runs/{run_id}/artifacts",
        files={
            "file": (
                "policy.md",
                b"# Retriever\n\nAnswers only from retrieved context and must cite the chunk.\n\n"
                b"Refunds are accepted within 30 days of purchase.\n",
                "text/markdown",
            )
        },
    )
    assert uploaded.status_code == 200

    analyzed = client.post(f"/api/runs/{run_id}/analyze")
    assert analyzed.status_code == 200
    assert analyzed.json()["phase"] == "awaiting_analysis_review"

    approved = client.post(f"/api/runs/{run_id}/analysis/approve")
    assert approved.json()["phase"] == "awaiting_plan_review"

    dataset = client.post(
        f"/api/runs/{run_id}/dataset",
        json={
            "cases": [
                {
                    "id": "c1",
                    "input": "What is the refund window?",
                    "expected": "30 days",
                    "context": ["Refunds are accepted within 30 days of purchase."],
                    "output": {"response": "Refunds are accepted within 30 days of purchase. [1]"},
                }
            ]
        },
    )
    assert dataset.status_code == 200

    finished = client.post(f"/api/runs/{run_id}/plan/approve")
    assert finished.status_code == 200
    body = finished.json()
    assert body["phase"] == "completed"
    assert body["aggregate"]["risk_level"] in {"low", "medium", "high", "critical"}
    assert len(body["chart_files"]) == 4

    report = client.get(f"/api/runs/{run_id}/report")
    assert report.status_code == 200
    assert "API RAG" in report.text
    chart = client.get(f"/api/runs/{run_id}/charts/{body['chart_files'][0]}")
    assert chart.status_code == 200
    assert "plotly" in chart.text.lower()

    listing = client.get("/api/runs")
    assert listing.json()[0]["id"] == run_id
    assert listing.json()[0]["overall"] is not None


def test_missing_run_is_a_404(tmp_path):
    client = _client(tmp_path)
    response = client.get("/api/runs/" + "a" * 32)
    assert response.status_code == 404
    assert "detail" in response.json()
