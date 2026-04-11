import { useState, useRef } from 'react'

const CLASS_NAMES = [
  'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [modelChoice, setModelChoice] = useState('CNN')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const fileInputRef = useRef(null)

  const handleFile = (e) => {
    const selected = e.target.files?.[0]
    if (selected) {
      setFile(selected)
      const objUrl = URL.createObjectURL(selected)
      setPreview(objUrl)
      setResult(null)
      setError(null)
    }
  }

  const handlePredict = async () => {
    if (!file) return

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('model_choice', modelChoice)

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`)
      }

      const data = await response.json()
      if (data.error) throw new Error(data.error)
      
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <header>
        <h1 className="hero-title">👗 Fashion-MNIST Discovery</h1>
        <p className="hero-subtitle">Classify clothing items using Deep Learning models</p>
      </header>

      <main className="dashboard">
        {/* Left Col: Upload & Controls */}
        <div className="glass-card left-col">
          <h2>Input Interface</h2>
          
          <div className="controls">
            {!preview ? (
              <div 
                className="upload-zone"
                onClick={() => fileInputRef.current?.click()}
              >
                <span className="icon">📤</span>
                <p>Click or drag image here<br/>(28x28 grayscale recommended)</p>
              </div>
            ) : (
              <div className="preview-container">
                <img src={preview} alt="Upload preview" />
                <br />
                <button 
                  style={{marginTop: '1rem', background: 'transparent', border:'1px solid #444', color:'white', padding:'0.5rem 1rem', borderRadius:'6px', cursor:'pointer'}}
                  onClick={() => { setFile(null); setPreview(null); setResult(null); }}
                >
                  Remove Image
                </button>
              </div>
            )}
            
            <input 
              type="file" 
              accept="image/png, image/jpeg" 
              className="hidden-input" 
              ref={fileInputRef}
              onChange={handleFile}
            />

            <label style={{display: 'block', marginBottom: '0.5rem', color: 'var(--text-muted)'}}>
              Select Architecture:
            </label>
            <select 
              value={modelChoice} 
              onChange={(e) => setModelChoice(e.target.value)}
            >
              <option value="CNN">CNN (Convolutional Neural Network)</option>
              <option value="ANN">ANN (Fully-Connected Network)</option>
            </select>

            <button 
              className="primary-btn"
              onClick={handlePredict}
              disabled={!file || loading}
            >
              {loading ? <span className="loader"></span> : '🔮 Generate Prediction'}
            </button>

            {error && (
              <div style={{marginTop: '1rem', padding: '1rem', background: 'rgba(255,82,82,0.1)', color: '#ff5252', borderRadius: '8px'}}>
                <strong>Error:</strong> {error}
                <br/><small>Make sure python backend is running via `uvicorn api:app`</small>
              </div>
            )}

          </div>
        </div>

        {/* Right Col: Results */}
        <div className="glass-card right-col">
          {result ? (
            <>
              <div className="result-header">
                <h2>{result.model} Prediction</h2>
                <div className="prediction-name">{result.top_prediction}</div>
                <div className="confidence">
                  Confidence: {(result.top_3[0].probability * 100).toFixed(1)}%
                </div>
              </div>

              <h3>Class Probabilities</h3>
              <div className="bar-chart" style={{marginTop: '1rem'}}>
                {result.confidence_distribution
                  .sort((a,b) => b.probability - a.probability)
                  .map((item, idx) => {
                    const isTop = idx === 0;
                    const pct = (item.probability * 100).toFixed(1);
                    return (
                      <div className={`bar-row ${isTop ? 'top' : ''}`} key={item.className}>
                        <div className="class-name">{item.className}</div>
                        <div className="bar-bg">
                          <div className="bar-fill" style={{ width: `${pct}%` }}></div>
                        </div>
                        <div className="pct">{pct}%</div>
                      </div>
                    )
                })}
              </div>
            </>
          ) : (
            <div style={{height: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', color: 'var(--text-muted)', textAlign: 'center'}}>
              <span style={{fontSize: '4rem', marginBottom: '1rem', opacity: 0.5}}>🔬</span>
              <p>Upload an image and run prediction to see results here.</p>
            </div>
          )}
        </div>
      </main>
    </>
  )
}

export default App
