import { useState, useRef, useEffect, useCallback } from 'react'

const API_ENDPOINT = 'http://127.0.0.1:8000/verify'

function UploadSlot({ label, hint, file, onSelect, accept = 'image/*' }) {
  const inputRef = useRef(null)
  const [isDragging, setIsDragging] = useState(false)
  const [previewUrl, setPreviewUrl] = useState(null)

  // Safely create and cleanup Blob URLs to prevent memory leaks
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return
    }

    const objectUrl = URL.createObjectURL(file)
    setPreviewUrl(objectUrl)

    // Revoke object URL from memory on cleanup
    return () => {
      URL.revokeObjectURL(objectUrl)
    }
  }, [file])

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault()
      setIsDragging(false)
      const dropped = e.dataTransfer.files?.[0]
      if (dropped) onSelect(dropped)
    },
    [onSelect]
  )

  return (
    <div className="slot">
      <div className="slot-label">
        <span>{label}</span>
        <span className="slot-hint">{hint}</span>
      </div>
      <button
        type="button"
        className={`slot-well ${isDragging ? 'is-dragging' : ''} ${file ? 'has-file' : ''}`}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        {previewUrl ? (
          <img src={previewUrl} alt={`${label} preview`} className="slot-preview" />
        ) : (
          <div className="slot-placeholder">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path
                d="M12 4v12m0-12 5 5m-5-5-5 5M5 20h14"
                stroke="currentColor"
                strokeWidth="1.4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <span>Choose or drop a photo</span>
          </div>
        )}
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        hidden
        onChange={(e) => {
          const picked = e.target.files?.[0]
          if (picked) onSelect(picked)
        }}
      />
    </div>
  )
}

function DataRow({ label, value }) {
  return (
    <div className="data-row">
      <span className="data-row-label">{label}</span>
      <span className="data-row-value">{value ?? '—'}</span>
    </div>
  )
}

function Stamp({ passed }) {
  return (
    <div className={`stamp ${passed ? 'stamp-pass' : 'stamp-fail'}`}>
      <svg viewBox="0 0 140 140" className="stamp-ring" aria-hidden="true">
        <circle cx="70" cy="70" r="64" fill="none" stroke="currentColor" strokeWidth="2" />
        <circle cx="70" cy="70" r="54" fill="none" stroke="currentColor" strokeWidth="1" />
      </svg>
      <div className="stamp-text">
        <span className="stamp-word">{passed ? 'VERIFIED' : 'NOT VERIFIED'}</span>
      </div>
    </div>
  )
}

export default function App() {
  const [idCard, setIdCard] = useState(null)
  const [selfie, setSelfie] = useState(null)
  const [status, setStatus] = useState('idle') // idle | loading | done | error
  const [result, setResult] = useState(null)
  const [errorMessage, setErrorMessage] = useState('')

  const canSubmit = idCard && selfie && status !== 'loading'

  async function handleSubmit() {
    setStatus('loading')
    setErrorMessage('')
    setResult(null)

    const formData = new FormData()
    formData.append('id_card', idCard)
    formData.append('selfie', selfie)

    try {
      const res = await fetch(API_ENDPOINT, { method: 'POST', body: formData })
      const payload = await res.json().catch(() => null)

      if (!res.ok) {
        throw new Error(payload?.detail || `Request failed with status ${res.status}`)
      }

      setResult(payload)
      setStatus('done')
    } catch (err) {
      setErrorMessage(err.message || 'Something went wrong while verifying.')
      setStatus('error')
    }
  }

  function handleReset() {
    setIdCard(null)
    setSelfie(null)
    setResult(null)
    setStatus('idle')
    setErrorMessage('')
  }

  const face = result?.face_matching_score
  const mrz = result?.mrz_viz_matching
  const overallPass = Boolean(face?.verified) && Boolean(mrz?.mrz_viz_verification?.verified)

  return (
    <div className="page">
      <div className="document">
        <header className="document-header">
          <div className="header-mark">ID·VF</div>
          <div className="header-text">
            <h1>Identity Verification</h1>
            <p>Match a government ID against a live photo and cross-check the printed record</p>
          </div>
        </header>

        <div className="document-body">
          <section className="leaf leaf-intake">
            <h2 className="leaf-title">01 — Submit documents</h2>

            <UploadSlot
              label="Identity document"
              hint="Passport or ID card, full page"
              file={idCard}
              onSelect={setIdCard}
            />
            <UploadSlot
              label="Live selfie"
              hint="Clear, front-facing, well lit"
              file={selfie}
              onSelect={setSelfie}
            />

            <div className="actions">
              <button
                type="button"
                className="btn-primary"
                disabled={!canSubmit}
                onClick={handleSubmit}
              >
                {status === 'loading' ? 'Verifying…' : 'Run verification'}
              </button>
              {(result || errorMessage) && (
                <button type="button" className="btn-ghost" onClick={handleReset}>
                  Start over
                </button>
              )}
            </div>

            {status === 'error' && (
              <div className="notice notice-error">
                <strong>Verification did not complete.</strong>
                <span>{errorMessage}</span>
              </div>
            )}
          </section>

          <div className="perforation" aria-hidden="true" />

          <section className="leaf leaf-result">
            <h2 className="leaf-title">02 — Result</h2>

            {status === 'idle' && (
              <p className="empty-state">
                Add both photos on the left, then run verification to see the printed result here.
              </p>
            )}

            {status === 'loading' && (
              <p className="empty-state">Reading the document and comparing faces…</p>
            )}

            {status === 'done' && result && (
              <div className="result">
                <Stamp passed={overallPass} />

                <div className="result-block">
                  <h3>Face match</h3>
                  <DataRow label="Similarity" value={face?.similarity_score} />
                  <DataRow label="Distance" value={face?.distance} />
                  <DataRow label="Threshold" value={face?.threshold} />
                  <DataRow label="Verified" value={face?.verified ? 'Yes' : 'No'} />
                </div>

                <div className="result-block">
                  <h3>MRZ / printed record</h3>
                  {mrz?.mrz_found ? (
                    <>
                      <DataRow label="MRZ line 1" value={mrz.mrz_line_1} />
                      <DataRow
                        label="Surname match"
                        value={mrz.mrz_viz_verification?.surname_match ? 'Yes' : 'No'}
                      />
                      <DataRow
                        label="Given name match"
                        value={mrz.mrz_viz_verification?.given_name_match ? 'Yes' : 'No'}
                      />
                      <DataRow
                        label="MRZ surname"
                        value={mrz.mrz_viz_verification?.extracted_mrz?.surname}
                      />
                      <DataRow
                        label="MRZ given names"
                        value={mrz.mrz_viz_verification?.extracted_mrz?.given_names}
                      />
                    </>
                  ) : (
                    <p className="empty-state">No MRZ line was found on the document.</p>
                  )}
                </div>
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  )
}