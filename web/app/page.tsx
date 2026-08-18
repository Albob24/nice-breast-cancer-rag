"use client";

import { FormEvent, useState } from "react";

const API_URL = "http://127.0.0.1:8000";

// =========================================================
// TYPES
// =========================================================

type Citation = {
  source: string;
  page: string | number;
};

type Source = {
  rank: number;
  source: string;
  page: string | number;
  section?: string;
  distance?: number;
  text?: string;
};

type RetrievalInfo = {
  top_k?: number;
  generation_context_k?: number;
  minimum_distance?: number;
  threshold?: number;
  vectors?: number;
  experiment?: string;
  collection?: string;
};

type ValidationInfo = {
  passed?: boolean;
  citation_found?: boolean;
  footer_found?: boolean;
};

type ChatResponse = {
  answer: string;
  status: string;
  citations?: Citation[];
  sources?: Source[];
  retrieval?: RetrievalInfo;
  validation?: ValidationInfo;
  latency_ms?: number;
};

type ApiError = {
  message: string;
  code?: number;
  type?: string;
  retryable?: boolean;
};

// =========================================================
// ERROR PARSER
// =========================================================

function extractApiError(
  response: Response,
  data: unknown
): ApiError {
  const fallback =
    "Unable to complete the request. Please try again.";

  if (!data || typeof data !== "object") {
    return {
      message: fallback,
      code: response.status,
      retryable: response.status >= 500,
    };
  }

  const payload = data as Record<string, unknown>;

  // FastAPI detail
  const detail = payload.detail;

  // detail is a string
  if (typeof detail === "string") {
    return {
      message: detail,
      code: response.status,
      retryable: response.status >= 500,
    };
  }

  // detail is an object
  if (detail && typeof detail === "object") {
    const detailObject = detail as Record<string, unknown>;

    const message =
      typeof detailObject.message === "string"
        ? detailObject.message
        : typeof detailObject.error === "string"
          ? detailObject.error
          : typeof detailObject.detail === "string"
            ? detailObject.detail
            : fallback;

    const errorType =
      typeof detailObject.type === "string"
        ? detailObject.type
        : undefined;

    return {
      message,
      code: response.status,
      type: errorType,
      retryable:
        response.status === 429 ||
        response.status === 502 ||
        response.status === 503 ||
        response.status === 504,
    };
  }

  // Generic message
  if (typeof payload.message === "string") {
    return {
      message: payload.message,
      code: response.status,
      retryable: response.status >= 500,
    };
  }

  return {
    message: fallback,
    code: response.status,
    retryable: response.status >= 500,
  };
}

// =========================================================
// STATUS HELPERS
// =========================================================

function getStatusLabel(status: string) {
  switch (status) {
    case "GENERATED":
      return "Generated";

    case "REJECTED":
      return "Insufficient evidence";

    case "QUOTA_EXCEEDED":
      return "AI quota exceeded";

    case "API_ERROR":
      return "API error";

    default:
      return status || "Unknown";
  }
}

function getStatusClass(status: string) {
  switch (status) {
    case "GENERATED":
      return "badge success";

    case "REJECTED":
      return "badge warning";

    case "QUOTA_EXCEEDED":
    case "API_ERROR":
      return "badge error";

    default:
      return "badge warning";
  }
}

// =========================================================
// COMPONENT
// =========================================================

export default function Home() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] =
    useState<ChatResponse | null>(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] =
    useState<ApiError | null>(null);

  const [selectedSource, setSelectedSource] =
    useState<Source | null>(null);

  // =======================================================
  // ASK QUESTION
  // =======================================================

  async function askQuestion(
    event: FormEvent<HTMLFormElement>
  ) {
    event.preventDefault();

    const trimmedQuestion =
      question.trim();

    if (!trimmedQuestion || loading) {
      return;
    }

    setLoading(true);
    setError(null);
    setResponse(null);
    setSelectedSource(null);

    try {
      const res = await fetch(
        `${API_URL}/api/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question: trimmedQuestion,
          }),
        }
      );

      let data: unknown = null;

      try {
        data = await res.json();
      } catch {
        data = null;
      }

      if (!res.ok) {
        throw extractApiError(
          res,
          data
        );
      }

      if (
        !data ||
        typeof data !== "object"
      ) {
        throw {
          message:
            "The backend returned an invalid response.",
          code: res.status,
        } satisfies ApiError;
      }

      setResponse(
        data as ChatResponse
      );
    } catch (err) {
      if (
        err &&
        typeof err === "object" &&
        "message" in err
      ) {
        setError(
          err as ApiError
        );
      } else if (err instanceof Error) {
        setError({
          message: err.message,
          retryable: true,
        });
      } else {
        setError({
          message:
            "Unable to connect to the NICE AI backend.",
          retryable: true,
        });
      }
    } finally {
      setLoading(false);
    }
  }

  // =======================================================
  // CLEAR CHAT
  // =======================================================

  function clearChat() {
    setQuestion("");
    setResponse(null);
    setError(null);
    setSelectedSource(null);
  }

  // =======================================================
  // QUICK QUESTION
  // =======================================================

  function useExample(text: string) {
    setQuestion(text);
    setError(null);
    setResponse(null);
    setSelectedSource(null);
  }

  // =======================================================
  // RENDER
  // =======================================================

  return (
    <main className="app-shell">

      {/* ==================================================
          SIDEBAR
      ================================================== */}

      <aside className="sidebar">

        <div className="brand">

          <div className="brand-mark">
            N
          </div>

          <div>
            <h1>NICE AI</h1>

            <p>
              Breast Cancer Assistant
            </p>
          </div>

        </div>

        <button
          className="new-chat-button"
          onClick={clearChat}
        >
          <span>+</span>
          New conversation
        </button>

        <div className="sidebar-section">

          <div className="sidebar-label">
            KNOWLEDGE BASE
          </div>

          <div className="guideline-card active">

            <div className="guideline-icon">
              N
            </div>

            <div>
              <strong>
                NG101
              </strong>

              <span>
                Early & locally advanced
              </span>
            </div>

          </div>

          <div className="guideline-card">

            <div className="guideline-icon secondary">
              C
            </div>

            <div>
              <strong>
                CG81
              </strong>

              <span>
                Advanced breast cancer
              </span>
            </div>

          </div>

        </div>

        <div className="sidebar-spacer" />

        <div className="system-status">

          <span className="status-dot" />

          <div>
            <strong>
              System online
            </strong>

            <span>
              RAG pipeline ready
            </span>
          </div>

        </div>

      </aside>

      {/* ==================================================
          MAIN PANEL
      ================================================== */}

      <section className="main-panel">

        {/* =================================================
            TOP BAR
        ================================================= */}

        <header className="topbar">

          <div>

            <div className="eyebrow">
              NICE CLINICAL GUIDELINES
            </div>

            <h2>
              Breast Cancer AI Assistant
            </h2>

          </div>

          <div className="topbar-status">

            <span className="status-dot" />

            Connected

          </div>

        </header>

        {/* =================================================
            CONTENT
        ================================================= */}

        <div className="content">

          {/* =================================================
              WELCOME
          ================================================= */}

          {!response &&
          !loading &&
          !error ? (

            <section className="welcome">

              <div className="welcome-icon">
                ✦
              </div>

              <h3>
                Evidence-grounded clinical information
              </h3>

              <p>
                Ask questions about breast cancer
                diagnosis, treatment and management
                using the NICE NG101 and CG81
                guidelines.
              </p>

              <div className="example-grid">

                <button
                  onClick={() =>
                    useExample(
                      "What endocrine therapy is recommended for postmenopausal women with ER-positive invasive breast cancer?"
                    )
                  }
                >
                  <span>
                    Endocrine therapy
                  </span>

                  <small>
                    Ask about treatment options
                    and recommendations
                  </small>

                </button>

                <button
                  onClick={() =>
                    useExample(
                      "What is neoadjuvant endocrine therapy used for in postmenopausal women with ER-positive invasive breast cancer?"
                    )
                  }
                >
                  <span>
                    Neoadjuvant therapy
                  </span>

                  <small>
                    Understand the role of
                    preoperative treatment
                  </small>

                </button>

                <button
                  onClick={() =>
                    useExample(
                      "What recommendations are given for HER2-positive breast cancer?"
                    )
                  }
                >
                  <span>
                    HER2-positive disease
                  </span>

                  <small>
                    Explore guideline-based
                    recommendations
                  </small>

                </button>

                <button
                  onClick={() =>
                    useExample(
                      "What information and psychological support should be provided to people with breast cancer?"
                    )
                  }
                >
                  <span>
                    Patient support
                  </span>

                  <small>
                    Information and psychological
                    support guidance
                  </small>

                </button>

              </div>

            </section>

          ) : null}

          {/* =================================================
              LOADING
          ================================================= */}

          {loading ? (

            <section className="loading-card">

              <div className="loader" />

              <h3>
                Analyzing the NICE guidelines
              </h3>

              <p>
                Retrieving relevant evidence
                and generating a grounded answer.
              </p>

            </section>

          ) : null}

          {/* =================================================
              PROFESSIONAL ERROR UI
          ================================================= */}

          {error ? (

            <section
              className={`error-card ${
                error.code === 429 ||
                error.code === 503
                  ? "quota-error"
                  : ""
              }`}
            >

              <div className="error-icon">
                !
              </div>

              <div className="error-content">

                <h3>
                  {error.code === 429 ||
                  error.code === 503
                    ? "AI generation temporarily unavailable"
                    : "Unable to complete the request"}
                </h3>

                <p>
                  {error.message}
                </p>

                {/* -----------------------------------------
                    SYSTEM STATUS
                ----------------------------------------- */}

                <div className="error-status-grid">

                  <div>
                    <span>
                      Retrieval system
                    </span>

                    <strong className="status-good">
                      ✓ Available
                    </strong>
                  </div>

                  <div>
                    <span>
                      Knowledge base
                    </span>

                    <strong className="status-good">
                      ✓ Available
                    </strong>
                  </div>

                  <div>
                    <span>
                      AI generation
                    </span>

                    <strong
                      className={
                        error.code === 429 ||
                        error.code === 503
                          ? "status-warning"
                          : "status-error"
                      }
                    >
                      {error.code === 429 ||
                      error.code === 503
                        ? "⚠ Temporarily limited"
                        : "⚠ Unavailable"}
                    </strong>
                  </div>

                </div>

                {/* -----------------------------------------
                    RETRY MESSAGE
                ----------------------------------------- */}

                {error.retryable ? (

                  <p className="error-hint">
                    The problem may be temporary.
                    Please wait and try the question
                    again.
                  </p>

                ) : null}

                {/* -----------------------------------------
                    RETRY BUTTON
                ----------------------------------------- */}

                <button
                  className="retry-button"
                  onClick={() => {
                    const form =
                      document.querySelector(
                        "form.composer"
                      ) as HTMLFormElement | null;

                    if (form) {
                      form.requestSubmit();
                    }
                  }}
                  disabled={
                    loading ||
                    !question.trim()
                  }
                >
                  Try again
                </button>

              </div>

            </section>

          ) : null}

          {/* =================================================
              ANSWER
          ================================================= */}

          {response ? (

            <section className="answer-layout">

              <div className="answer-column">

                {/* -------------------------------------------
                    QUESTION
                ------------------------------------------- */}

                <div className="question-bubble">

                  <span>
                    You asked
                  </span>

                  <p>
                    {question}
                  </p>

                </div>

                {/* -------------------------------------------
                    ANSWER CARD
                ------------------------------------------- */}

                <article className="answer-card">

                  <div className="answer-header">

                    <div className="answer-title">

                      <div className="ai-avatar">
                        ✦
                      </div>

                      <div>

                        <strong>
                          NICE AI
                        </strong>

                        <span>
                          Evidence-grounded response
                        </span>

                      </div>

                    </div>

                    <div
                      className={getStatusClass(
                        response.status
                      )}
                    >
                      {getStatusLabel(
                        response.status
                      )}
                    </div>

                  </div>

                  <div className="answer-body">

                    <div className="answer-text">
                      {response.answer}
                    </div>

                  </div>

                  {/* -----------------------------------------
                      CITATIONS
                  ----------------------------------------- */}

                  {response.citations &&
                  response.citations.length > 0 ? (

                    <div className="citation-section">

                      <div className="section-heading">

                        <span>
                          REFERENCES
                        </span>

                        <small>
                          {
                            response.citations.length
                          }{" "}
                          citation
                          {response.citations.length === 1
                            ? ""
                            : "s"}
                        </small>

                      </div>

                      <div className="citation-list">

                        {response.citations.map(
                          (
                            citation,
                            index
                          ) => (

                            <button
                              key={`${citation.source}-${citation.page}-${index}`}
                              onClick={() => {

                                const source =
                                  response.sources?.find(
                                    (item) =>
                                      item.source ===
                                        citation.source &&
                                      String(
                                        item.page
                                      ) ===
                                        String(
                                          citation.page
                                        )
                                  );

                                if (source) {
                                  setSelectedSource(
                                    source
                                  );
                                }

                              }}
                              className="citation-chip"
                            >

                              <span>
                                {citation.source}
                              </span>

                              <span>
                                Page{" "}
                                {citation.page}
                              </span>

                            </button>

                          )
                        )}

                      </div>

                    </div>

                  ) : null}

                </article>

                {/* -------------------------------------------
                    VALIDATION
                ------------------------------------------- */}

                {response.validation ? (

                  <div className="validation-bar">

                    <span
                      className={
                        response.validation.passed
                          ? "validation-icon valid"
                          : "validation-icon"
                      }
                    >
                      {response.validation.passed
                        ? "✓"
                        : "!"}
                    </span>

                    <div>

                      <strong>
                        {response.validation.passed
                          ? "Answer validation passed"
                          : "Answer validation requires review"}
                      </strong>

                      <span>
                        Citation and grounding
                        checks were evaluated
                        by the backend.
                      </span>

                    </div>

                  </div>

                ) : null}

              </div>

              {/* =================================================
                  EVIDENCE COLUMN
              ================================================= */}

              <aside className="evidence-column">

                <div className="evidence-header">

                  <div>

                    <div className="eyebrow">
                      RETRIEVAL
                    </div>

                    <h3>
                      Evidence sources
                    </h3>

                  </div>

                  <span className="source-count">
                    {response.sources?.length || 0}
                  </span>

                </div>

                {/* ---------------------------------------------
                    SOURCE LIST
                --------------------------------------------- */}

                <div className="source-list">

                  {response.sources?.map(
                    (source) => (

                      <button
                        key={`${source.rank}-${source.source}-${source.page}`}
                        className={`source-card ${
                          selectedSource === source
                            ? "selected"
                            : ""
                        }`}
                        onClick={() =>
                          setSelectedSource(
                            source
                          )
                        }
                      >

                        <div className="source-top">

                          <span className="rank">
                            #{source.rank}
                          </span>

                          <span className="distance">
                            {source.distance !==
                            undefined
                              ? source.distance.toFixed(
                                  3
                                )
                              : "—"}
                          </span>

                        </div>

                        <div className="source-name">
                          {source.source}
                        </div>

                        <div className="source-page">
                          Page {source.page}
                        </div>

                        {source.section ? (

                          <div className="source-section">
                            {source.section}
                          </div>

                        ) : null}

                      </button>

                    )
                  )}

                </div>

                {/* ---------------------------------------------
                    EVIDENCE DETAIL
                --------------------------------------------- */}

                {selectedSource ? (

                  <div className="evidence-detail">

                    <div className="detail-heading">

                      <div>

                        <span>
                          {selectedSource.source}
                          {" · "}
                          Page{" "}
                          {selectedSource.page}
                        </span>

                        <h4>
                          Retrieved evidence
                        </h4>

                      </div>

                      <button
                        onClick={() =>
                          setSelectedSource(
                            null
                          )
                        }
                        aria-label="Close evidence"
                      >
                        ×
                      </button>

                    </div>

                    {selectedSource.section ? (

                      <div className="detail-section">

                        <span>
                          SECTION
                        </span>

                        <p>
                          {selectedSource.section}
                        </p>

                      </div>

                    ) : null}

                    <div className="detail-text">

                      {selectedSource.text ||
                        "Evidence text was not included in the API response."}

                    </div>

                  </div>

                ) : null}

                {/* ---------------------------------------------
                    RETRIEVAL METRICS
                --------------------------------------------- */}

                {response.retrieval ? (

                  <div className="retrieval-metrics">

                    <div className="section-heading">

                      <span>
                        RAG METRICS
                      </span>

                    </div>

                    <div className="metrics-grid">

                      <div>

                        <span>
                          Experiment
                        </span>

                        <strong>
                          {response.retrieval
                            .experiment ||
                            "—"}
                        </strong>

                      </div>

                      <div>

                        <span>
                          Vectors
                        </span>

                        <strong>
                          {response.retrieval
                            .vectors ??
                            "—"}
                        </strong>

                      </div>

                      <div>

                        <span>
                          Top-K
                        </span>

                        <strong>
                          {response.retrieval
                            .top_k ??
                            "—"}
                        </strong>

                      </div>

                      <div>

                        <span>
                          Context
                        </span>

                        <strong>
                          {response.retrieval
                            .generation_context_k ??
                            "—"}
                        </strong>

                      </div>

                    </div>

                  </div>

                ) : null}

              </aside>

            </section>

          ) : null}

        </div>

        {/* =================================================
            COMPOSER
        ================================================= */}

        <div className="composer-wrapper">

          <form
            className="composer"
            onSubmit={askQuestion}
          >

            <textarea
              value={question}
              onChange={(event) =>
                setQuestion(
                  event.target.value
                )
              }
              placeholder="Ask a question about the NICE breast cancer guidelines..."
              rows={1}
              disabled={loading}
              onKeyDown={(event) => {

                if (
                  event.key === "Enter" &&
                  !event.shiftKey
                ) {

                  event.preventDefault();

                  const form =
                    event.currentTarget.form;

                  if (form) {
                    form.requestSubmit();
                  }

                }

              }}
            />

            <button
              type="submit"
              disabled={
                loading ||
                !question.trim()
              }
              className="send-button"
            >
              {loading
                ? "..."
                : "Ask"}
            </button>

          </form>

          <div className="composer-footer">

            <span>
              Answers are grounded only in
              the retrieved NICE guideline
              evidence.
            </span>

            <span>
              NG101 · CG81
            </span>

          </div>

        </div>

      </section>

    </main>
  );
}