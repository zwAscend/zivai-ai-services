import os
from flask import Flask, jsonify
from dotenv import load_dotenv
from asag_engine.db import init_db

from .agent_routes import bp as agent_bp
from .asag_routes import bp as asag_bp
from .dev_routes import bp as dev_bp
from .dkt_routes import bp as dkt_bp
from .grading_routes import bp as grading_bp
from .health_routes import bp as health_bp
from .llm_routes import bp as llm_bp
from .ocr_routes import bp as ocr_bp


load_dotenv()
def create_app() -> Flask:
    load_dotenv()
    app = Flask(__name__)

    auto_create = os.getenv("AUTO_CREATE_TABLES", "false").lower() == "true"
    init_db(auto_create=auto_create)

    app.register_blueprint(dev_bp)
    app.register_blueprint(dkt_bp)
    app.register_blueprint(agent_bp)
    app.register_blueprint(asag_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(grading_bp)
    app.register_blueprint(llm_bp)
    app.register_blueprint(ocr_bp)

    @app.get("/")
    def root():
        return jsonify({
            "name": "ZivAI AI Services Engine (MindSpore/MindNLP)",
            "status": "ok",
            "mode": "shared-schema rebuild",
            "endpoints": [
                "/api/v1/health",
                "/api/v1/dev/bootstrap-grading-scenario [POST]",
                "/api/v1/dev/grading-targets [GET]",
                "/api/v1/dkt/update [POST]",
                "/api/v1/dkt/mastery/<student_id> [GET]",
                "/api/v1/asag/score [POST]",
                "/api/v1/agents/teacher/assessment-generation [POST]",
                "/api/v1/agents/teacher/plan-generation [POST]",
                "/api/v1/agents/student/assessment [POST]",
                "/api/v1/agents/ocr/general [POST]",
                "/api/v1/grade/question [POST]",
                "/api/v1/grade/assessment [POST]",
                "/api/v1/grade/attempt-answer/<attempt_answer_id> [POST]",
                "/api/v1/grade/assessment-attempt/<assessment_attempt_id> [POST]",
                "/api/v1/llm/test [POST]",
            ],
            "notes": [
                "Deprecated standalone ASAG CRUD routes have been removed.",
                "Shared-core-db grading and other AI workflows are being rebuilt against the lms.* and ai.* contracts.",
                "Grading now reads from shared LMS attempts/answers and writes back AI scoring, feedback, and inference traces."
            ],
        })

    return app

if __name__ == "__main__":
    app = create_app()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug, use_reloader=False)
