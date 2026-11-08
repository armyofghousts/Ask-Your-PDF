import config  
from flask import Flask, render_template, request

from rag import DONT_KNOW, answer_question

app = Flask(__name__)


@app.get("/")
@app.post("/")
def index():
    question = ""
    answer = None
    sources = []
    error = None
    if request.method == "POST":
        question = request.form.get("question", "")
        result = answer_question(question)
        answer = result["answer"]
        sources = result["sources"]
        error = result["error"]
    return render_template(
        "index.html",
        question=question,
        answer=answer,
        sources=sources,
        error=error,
        dont_know=DONT_KNOW,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
