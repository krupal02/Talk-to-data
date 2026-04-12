// Copyright 2026 Talk-to-Data Contributors
// Licensed under the Apache License, Version 2.0

/**
 * App.jsx – root application component.
 *
 * Manages top-level state (session, query results) and delegates
 * all rendering to child components. No business logic lives here.
 */

import { useState, useCallback } from 'react';
import FileUpload from './components/FileUpload';
import QueryInput from './components/QueryInput';
import AnswerCard from './components/AnswerCard';
import useQuery from './hooks/useQuery';

export default function App() {
  const [session, setSession] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const { result, loading, error, ask, reset } = useQuery();

  const handleUploadSuccess = useCallback((uploadData) => {
    setSession(uploadData);
    reset();
  }, [reset]);

  const handleQuery = useCallback((question) => {
    if (!session) return;
    setCurrentQuestion(question);
    ask(session.session_id, question);
  }, [session, ask]);

  const handleNewDataset = useCallback(() => {
    setSession(null);
    setCurrentQuestion('');
    reset();
  }, [reset]);

  return (
    <div className="min-h-screen gradient-glow">
      {/* Header */}
      <header className="border-b border-surface-200/5">
        <div className="max-w-5xl mx-auto px-6 py-5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl gradient-primary flex items-center justify-center shadow-lg shadow-primary-500/20">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6" />
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-bold text-surface-100 tracking-tight">Talk to Data</h1>
              <p className="text-[11px] text-surface-200/40 font-medium">Self-Service Intelligence</p>
            </div>
          </div>

          {session && (
            <button
              id="new-dataset-btn"
              onClick={handleNewDataset}
              className="text-xs text-surface-200/50 hover:text-primary-300 
                         border border-surface-200/10 hover:border-primary-400/30 
                         px-3 py-1.5 rounded-lg transition-all duration-200"
            >
              New dataset
            </button>
          )}
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-5xl mx-auto px-6 py-10">
        {!session ? (
          /* Upload stage */
          <div className="flex flex-col items-center gap-8">
            <div className="text-center max-w-lg animate-fade-in">
              <h2 className="text-3xl font-bold text-surface-100 mb-3 tracking-tight">
                Ask your data anything
              </h2>
              <p className="text-surface-200/50 leading-relaxed">
                Upload a CSV or SQLite file, then ask questions in plain English.
                Get clear answers with charts and full source transparency.
              </p>
            </div>

            <div className="w-full max-w-xl">
              <FileUpload onUploadSuccess={handleUploadSuccess} />
            </div>

            {/* Feature pills */}
            <div className="flex flex-wrap justify-center gap-3 mt-2 animate-fade-in">
              {[
                { icon: '✦', label: 'Plain-language answers' },
                { icon: '◐', label: 'Auto-visualisation' },
                { icon: '⊘', label: 'PII auto-stripped' },
                { icon: '⬡', label: 'Source transparency' },
              ].map(({ icon, label }) => (
                <span
                  key={label}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-full 
                             bg-surface-800/40 border border-surface-200/5 
                             text-xs text-surface-200/40"
                >
                  <span className="text-primary-400">{icon}</span>
                  {label}
                </span>
              ))}
            </div>
          </div>
        ) : (
          /* Query stage */
          <div className="flex flex-col gap-6">
            <QueryInput
              onSubmit={handleQuery}
              loading={loading}
              disabled={!session}
            />

            {/* Loading state */}
            {loading && (
              <div className="glass rounded-2xl p-8 animate-fade-in">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-xl bg-primary-500/10 flex items-center justify-center">
                    <div className="w-5 h-5 border-2 border-primary-400/30 border-t-primary-400 rounded-full animate-spin" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-surface-100">Analysing your data…</p>
                    <p className="text-xs text-surface-200/40 mt-0.5">
                      Parsing intent → Running aggregations → Generating insights
                    </p>
                  </div>
                </div>
                <div className="mt-4 space-y-2.5">
                  <div className="h-3 w-full shimmer rounded-lg" />
                  <div className="h-3 w-4/5 shimmer rounded-lg" />
                  <div className="h-3 w-3/5 shimmer rounded-lg" />
                </div>
              </div>
            )}

            {/* Error state */}
            {error && (
              <div id="query-error" className="p-5 rounded-xl bg-red-500/10 border border-red-500/20 animate-slide-up">
                <div className="flex items-start gap-3">
                  <svg className="w-5 h-5 text-red-400 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                  <div>
                    <p className="text-sm font-medium text-red-300">Something went wrong</p>
                    <p className="text-xs text-red-300/70 mt-1">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Result */}
            {result && !loading && (
              <AnswerCard result={result} question={currentQuestion} />
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-surface-200/5 mt-auto">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between text-[11px] text-surface-200/30">
          <span>Talk to Data · NatWest Code for Purpose 2026</span>
          <span>Powered by Gemini AI</span>
        </div>
      </footer>
    </div>
  );
}
