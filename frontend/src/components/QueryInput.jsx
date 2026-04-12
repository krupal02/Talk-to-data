/**
 * QueryInput – text input for natural-language questions.
 *
 * Shows example queries as placeholders and handles submit on Enter
 * or button click. Disabled while a query is in flight.
 */

import { useState, useCallback } from 'react';

const EXAMPLE_QUERIES = [
  'Why did revenue drop last month?',
  'Compare North vs South region sales',
  'What makes up total revenue?',
  'Give me a weekly summary for orders',
];

export default function QueryInput({ onSubmit, loading, disabled }) {
  const [question, setQuestion] = useState('');
  const [placeholderIdx, setPlaceholderIdx] = useState(0);

  const handleSubmit = useCallback((e) => {
    e?.preventDefault();
    const q = question.trim();
    if (!q || loading || disabled) return;
    onSubmit(q);
  }, [question, loading, disabled, onSubmit]);

  const handleExampleClick = useCallback((example) => {
    setQuestion(example);
  }, []);

  return (
    <div className="w-full animate-fade-in">
      <form onSubmit={handleSubmit} className="relative">
        <div className="glass rounded-2xl p-2 flex items-center gap-2 transition-all focus-within:border-primary-400/40 focus-within:shadow-lg focus-within:shadow-primary-500/10">
          {/* Search icon */}
          <div className="pl-4">
            <svg className="w-5 h-5 text-surface-200/40" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z" />
            </svg>
          </div>

          <input
            id="query-input"
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={EXAMPLE_QUERIES[placeholderIdx % EXAMPLE_QUERIES.length]}
            onFocus={() => setPlaceholderIdx((i) => i + 1)}
            disabled={loading || disabled}
            className="flex-1 bg-transparent text-surface-100 placeholder-surface-200/30 outline-none py-3 text-base"
            aria-label="Ask a question about your data"
          />

          <button
            id="query-submit-btn"
            type="submit"
            disabled={!question.trim() || loading || disabled}
            className="px-6 py-3 rounded-xl gradient-primary text-white font-semibold text-sm 
                       disabled:opacity-30 disabled:cursor-not-allowed
                       hover:shadow-lg hover:shadow-primary-500/25 transition-all duration-200
                       active:scale-95"
          >
            {loading ? (
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Thinking…</span>
              </div>
            ) : (
              'Ask'
            )}
          </button>
        </div>
      </form>

      {/* Example queries */}
      {!disabled && (
        <div className="flex flex-wrap gap-2 mt-3">
          {EXAMPLE_QUERIES.map((example) => (
            <button
              key={example}
              onClick={() => handleExampleClick(example)}
              className="px-3 py-1.5 rounded-lg text-xs text-surface-200/50 
                         border border-surface-200/10 hover:border-primary-400/30 
                         hover:text-primary-300 transition-all duration-200"
            >
              {example}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
