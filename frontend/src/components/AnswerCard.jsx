/**
 * AnswerCard – displays the AI-generated answer with chart and source badges.
 *
 * Layout:
 *   1. Plain-language answer text (large, readable font)
 *   2. Source and metric badges
 *   3. ChartRenderer component
 *   4. Confidence note in muted text
 */

import ChartRenderer from './ChartRenderer';
import SourceBadge from './SourceBadge';

export default function AnswerCard({ result, question }) {
  if (!result) return null;

  const { answer, chart_type, chart_data, source_ref, metric_used, confidence_note } = result;

  return (
    <div id="answer-card" className="w-full glass rounded-2xl overflow-hidden animate-slide-up">
      {/* Answer section */}
      <div className="p-6 pb-4">
        {/* Intent indicator */}
        <div className="flex items-center gap-2 mb-4">
          <div className="w-2 h-2 rounded-full bg-primary-400 animate-pulse-soft" />
          <span className="text-xs font-medium text-primary-300 uppercase tracking-wider">
            AI Analysis
          </span>
        </div>

        {/* Answer text */}
        <p id="answer-text" className="text-base leading-relaxed text-surface-100/90 font-light">
          {answer}
        </p>

        {/* Badges */}
        <div className="flex flex-wrap items-center gap-2 mt-4">
          <SourceBadge
            sourceRef={source_ref}
            metricUsed={metric_used}
            confidenceNote={confidence_note}
          />
          <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
            <svg className="w-3 h-3 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <span className="text-xs text-emerald-300">{metric_used}</span>
          </span>
        </div>
      </div>

      {/* Chart section */}
      {chart_data && chart_data.length > 0 && (
        <div className="px-6 pb-2">
          <div className="rounded-xl bg-surface-800/30 p-4">
            <ChartRenderer
              chartType={chart_type}
              chartData={chart_data}
              title={question}
            />
          </div>
        </div>
      )}

      {/* Confidence note */}
      {confidence_note && (
        <div className="px-6 py-3 border-t border-surface-200/5">
          <p id="confidence-note" className="text-xs text-surface-200/40 flex items-center gap-1.5">
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
            {confidence_note}
          </p>
        </div>
      )}
    </div>
  );
}
