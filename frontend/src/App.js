import React, { useState, useEffect, createContext, useContext } from "react";
import "./App.css";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import axios from "axios";
import { mockData } from "./mockData";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

// Auth Context
const AuthContext = createContext();

const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');
    if (token && userData) {
      setUser(JSON.parse(userData));
    }
    setLoading(false);
  }, []);

  const login = (userData, token) => {
    localStorage.setItem('token', token);
    localStorage.setItem('user', JSON.stringify(userData));
    setUser(userData);
  };

  const logout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Components
const Login = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const { login } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    // Mock authentication for demo
    setTimeout(() => {
      if (username && password) {
        const userData = {
          user_id: "mock_user_123",
          username: username,
          api_key: "sk-docubrain-" + Math.random().toString(36).substr(2, 20)
        };
        login(userData, "mock_token_" + Math.random().toString(36));
      } else {
        setError('Please enter username and password');
      }
      setLoading(false);
    }, 1000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
      <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">DocuBrain</h1>
          <p className="text-gray-600">Turn your documents into a private knowledge base</p>
        </div>
        
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Username
            </label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>

          {error && (
            <div className="text-red-600 text-sm text-center">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition duration-200"
          >
            {loading ? 'Processing...' : (isLogin ? 'Login' : 'Register')}
          </button>
        </form>

        <div className="mt-6 text-center">
          <button
            onClick={() => setIsLogin(!isLogin)}
            className="text-blue-600 hover:text-blue-800 text-sm"
          >
            {isLogin ? "Don't have an account? Register" : "Already have an account? Login"}
          </button>
        </div>
      </div>
    </div>
  );
};

const Dashboard = () => {
  const { user, logout } = useAuth();
  const [documents, setDocuments] = useState(mockData.documents);
  const [uploading, setUploading] = useState(false);
  const [query, setQuery] = useState('');
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState([]);
  const [querying, setQuerying] = useState(false);

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.pdf')) {
      alert('Please select a PDF file');
      return;
    }

    setUploading(true);
    
    // Mock file upload
    setTimeout(() => {
      const newDoc = {
        id: Date.now(),
        filename: file.name,
        upload_time: new Date().toISOString(),
        chunk_count: Math.floor(Math.random() * 50) + 10,
        status: 'completed'
      };
      setDocuments([...documents, newDoc]);
      alert('Document uploaded and processed successfully!');
      setUploading(false);
      e.target.value = '';
    }, 2000);
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setQuerying(true);
    
    // Mock query processing
    setTimeout(() => {
      const mockAnswer = mockData.sampleAnswers[Math.floor(Math.random() * mockData.sampleAnswers.length)];
      const mockSources = documents.slice(0, 2).map((doc, index) => ({
        filename: doc.filename,
        chunk_index: Math.floor(Math.random() * doc.chunk_count),
        relevance_score: 0.8 + (Math.random() * 0.2)
      }));
      
      setAnswer(mockAnswer);
      setSources(mockSources);
      setQuerying(false);
    }, 1500);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">DocuBrain Dashboard</h1>
              <p className="text-sm text-gray-500">Welcome, {user.username}</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="bg-blue-50 px-4 py-2 rounded-lg">
                <p className="text-xs text-gray-600">Your API Key:</p>
                <p className="text-sm font-mono text-blue-600">{user.api_key}</p>
              </div>
              <button
                onClick={logout}
                className="bg-red-500 text-white px-4 py-2 rounded-lg hover:bg-red-600 transition duration-200"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Document Management */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-6">Document Management</h2>
            
            {/* Upload Section */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload PDF Document
              </label>
              <input
                type="file"
                accept=".pdf"
                onChange={handleFileUpload}
                disabled={uploading}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              {uploading && (
                <p className="text-blue-600 text-sm mt-2">Processing document...</p>
              )}
            </div>

            {/* Documents List */}
            <div>
              <h3 className="text-lg font-medium text-gray-700 mb-4">Your Documents</h3>
              <div className="space-y-3">
                {documents.length === 0 ? (
                  <p className="text-gray-500 text-center py-4">No documents uploaded yet</p>
                ) : (
                  documents.map((doc) => (
                    <div key={doc.id} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex justify-between items-start">
                        <div>
                          <h4 className="font-medium text-gray-800">{doc.filename}</h4>
                          <p className="text-sm text-gray-500">
                            {new Date(doc.upload_time).toLocaleDateString()} • {doc.chunk_count} chunks
                          </p>
                        </div>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          doc.status === 'completed'
                            ? 'bg-green-100 text-green-800'
                            : doc.status === 'processing'
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {doc.status}
                        </span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Query Section */}
          <div className="bg-white rounded-xl shadow-lg p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-6">Ask Questions</h2>
            
            <form onSubmit={handleQuery} className="mb-6">
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Your Question
                </label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask anything about your uploaded documents..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                  rows="3"
                  required
                />
              </div>
              <button
                type="submit"
                disabled={querying || documents.length === 0}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 transition duration-200"
              >
                {querying ? 'Searching...' : 'Ask Question'}
              </button>
            </form>

            {/* Answer Section */}
            {answer && (
              <div className="border-t pt-6">
                <h3 className="text-lg font-medium text-gray-700 mb-4">Answer</h3>
                <div className="bg-gray-50 rounded-lg p-4 mb-4">
                  <p className="text-gray-800 whitespace-pre-wrap">{answer}</p>
                </div>
                {sources.length > 0 && (
                  <div>
                    <h4 className="text-sm font-medium text-gray-600 mb-2">Sources:</h4>
                    <div className="space-y-1">
                      {sources.map((source, index) => (
                        <div key={index} className="text-xs text-gray-500">
                          📄 {source.filename} (chunk {source.chunk_index + 1}) - {Math.round(source.relevance_score * 100)}% relevant
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* API Integration Section */}
        <div className="mt-8 bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">API Integration</h2>
          <p className="text-gray-600 mb-4">
            Use your API key to integrate DocuBrain with your applications:
          </p>
          <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm overflow-x-auto">
            <div className="mb-2">
              <span className="text-gray-500"># External API Endpoint</span>
            </div>
            <div className="mb-2">
              <span className="text-blue-400">POST</span> {API}/external/query
            </div>
            <div className="mb-4">
              <span className="text-gray-500"># Form data:</span>
            </div>
            <div className="mb-1">api_key: <span className="text-yellow-400">{user.api_key}</span></div>
            <div>question: <span className="text-yellow-400">"Your question here"</span></div>
          </div>
        </div>
      </div>
    </div>
  );
};

const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth();

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  return user ? children : <Navigate to="/login" />;
};

function App() {
  return (
    <div className="App">
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/" element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            } />
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </div>
  );
}

export default App;