import React, { useMemo, useState } from 'react'
import axios from 'axios'
import { v4 as uuidv4 } from 'uuid'


const BASE_URL = 'http://localhost:8000'

async function postJson(url, body) {
  const res = await axios.post(url, body, { headers: { 'Content-Type': 'application/json' } })
  return res.data
}

export default function App() {
  const [sessionId] = useState(uuidv4())
  const [query, setQuery] = useState('')
  const [intermediateAnswer, setIntermediateAnswer] = useState('')
  const [feedback, setFeedback] = useState('')
  const [finalAnswer, setFinalAnswer] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const colors = useMemo(() => ({
    intermediate: '#e0ecff', 
    refined: '#ffeacc',      
    final: '#dcfce7',       
  }), [])

  const handleAsk = async () => {
    setError('')
    setFinalAnswer('')
    setIntermediateAnswer('')
    if (!query.trim()) return
    setLoading(true)
    try {
      const data = await postJson(`${BASE_URL}/ask`, { session_id: sessionId, query })
      setIntermediateAnswer(data?.intermediate_answer || '')
    } catch (e) {
      setError(e?.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleRefine = async () => {
    setError('')
    if (!intermediateAnswer) return
    if (!feedback.trim()) return
    setLoading(true)
    try {
      const data = await postJson(`${BASE_URL}/refine`, { session_id: sessionId, feedback })
      setIntermediateAnswer(data?.refined_intermediate_answer || '')
      setFeedback('')
    } catch (e) {
      setError(e?.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleGetFinal = async () => {
    setError('')
    if (!query.trim()) return
    setLoading(true)
    try {
      const data = await postJson(`${BASE_URL}/final`, { session_id: sessionId, query })
      setFinalAnswer(data?.final_answer || '')
      setIntermediateAnswer('')
      setFeedback('')
      setQuery('')
    } catch (e) {
      setError(e?.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <h1 style={styles.title}>Math Chatbot (HITL Demo)</h1>
        <p style={styles.subtitle}>Validate and refine intermediate answers before getting the final one.</p>

        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>Query</h2>
          <div style={{ display: 'flex', gap: 8 }}>
            <input
              style={styles.input}
              placeholder="Type your math question..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={loading}
            />
            <button style={styles.button} onClick={handleAsk} disabled={loading || !query.trim()}>
              {loading ? 'Asking...' : 'Ask'}
            </button>
          </div>
          <small>Session: {sessionId}</small>
        </section>

        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>Intermediate Answer (HITL)</h2>
          <div style={{ ...styles.answerBox, background: colors.intermediate }}>
            {intermediateAnswer ? (
              <pre style={styles.pre}>{intermediateAnswer}</pre>
            ) : (
              <span style={{ color: '#6b7280' }}>Ask a question to see the intermediate answer here.</span>
            )}
          </div>

          <label style={styles.label}>Your feedback or correction:</label>
          <textarea
            style={styles.textarea}
            placeholder="Point out mistakes, request clarity, or provide the correct steps."
            value={feedback}
            onChange={(e) => setFeedback(e.target.value)}
            disabled={loading || !intermediateAnswer}
          />
          <button style={{ ...styles.button, background: '#fb923c' }} onClick={handleRefine} disabled={loading || !feedback.trim() || !intermediateAnswer}>
            {loading ? 'Refining...' : 'Refine Answer'}
          </button>
          <p style={{ fontSize: 12, color: '#6b7280', marginTop: 6 }}>Validate the intermediate answer first. Loop refinement until correct.</p>
        </section>

        <section style={styles.section}>
          <h2 style={styles.sectionTitle}>Final Answer</h2>
          <div style={{ ...styles.answerBox, background: colors.final }}>
            {finalAnswer ? (
              <pre style={styles.pre}>{finalAnswer}</pre>
            ) : (
              <span style={{ color: '#6b7280' }}>Once satisfied, click Get Final Answer.</span>
            )}
          </div>
          <button style={{ ...styles.button, background: '#22c55e' }} onClick={handleGetFinal} disabled={loading || !query.trim()}>
            {loading ? 'Fetching...' : 'Get Final Answer'}
          </button>
        </section>

        {error && (
          <div style={styles.error}>Error: {error}</div>
        )}
      </div>
    </div>
  )
}

const styles = {
  page: {
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
  },
  container: {
    width: '100%',
    maxWidth: 900,
    background: '#fff',
    borderRadius: 16,
    boxShadow: '0 20px 40px rgba(0,0,0,0.15)',
    padding: 20,
  },
  title: {
    margin: 0,
    fontSize: 28,
  },
  subtitle: {
    marginTop: 6,
    marginBottom: 16,
    color: '#6b7280',
  },
  section: {
    marginTop: 16,
  },
  sectionTitle: {
    margin: 0,
    marginBottom: 8,
    fontSize: 18,
  },
  input: {
    flex: 1,
    padding: 12,
    border: '2px solid #e5e7eb',
    borderRadius: 10,
    outline: 'none',
    fontSize: 16,
  },
  textarea: {
    width: '100%',
    minHeight: 100,
    padding: 12,
    border: '2px solid #e5e7eb',
    borderRadius: 10,
    outline: 'none',
    fontSize: 14,
    resize: 'vertical',
    marginTop: 8,
    marginBottom: 8,
  },
  button: {
    padding: '12px 16px',
    background: '#6366f1',
    color: 'white',
    border: 'none',
    borderRadius: 10,
    cursor: 'pointer',
    fontWeight: 600,
  },
  answerBox: {
    width: '100%',
    minHeight: 100,
    borderRadius: 12,
    border: '1px solid #e5e7eb',
    padding: 12,
  },
  pre: {
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    margin: 0,
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
  },
  label: {
    display: 'block',
    marginTop: 8,
    fontSize: 14,
    color: '#374151',
  },
  error: {
    marginTop: 16,
    padding: 12,
    background: '#fee2e2',
    color: '#b91c1c',
    borderRadius: 10,
    border: '1px solid #fecaca',
  },
}


