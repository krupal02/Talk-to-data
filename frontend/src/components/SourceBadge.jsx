/**
 * SourceBadge – displays a provenance pill with metric definition modal.
 *
 * Shows "Based on [source_ref]" in a muted badge. Clicking opens a
 * small modal with the metric definition from the semantic layer.
 */

import { useState, useCallback } from 'react';

export default function SourceBadge({ sourceRef, metricUsed, confidenceNote }) {
  const [showModal, setShowModal] = useState(false);

  const toggleModal = useCallback(() => {
    setShowModal((prev) => !prev);
  }, []);

  return (
    <div className="relative inline-block">
      {/* Badge pill */}
      <button
        id="source-badge-btn"
        onClick={toggleModal}
        className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full 
                   bg-surface-700/50 border border-surface-200/10 
                   hover:border-primary-400/30 hover:bg-surface-700/70
                   transition-all duration-200 cursor-pointer group"
        aria-label="View source details"
      >
        <svg className="w-3.5 h-3.5 text-primary-400 group-hover:text-primary-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span className="text-xs text-surface-200/60 group-hover:text-surface-200/80">
          Based on {sourceRef}
        </span>
      </button>

      {/* Modal overlay */}
      {showModal && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={toggleModal}
            aria-hidden="true"
          />
          <div
            id="source-modal"
            className="absolute left-0 bottom-full mb-2 z-50 w-72 glass rounded-xl p-4 
                       shadow-xl shadow-black/30 animate-slide-up"
          >
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-semibold text-surface-100">Source Details</h4>
              <button
                onClick={toggleModal}
                className="text-surface-200/40 hover:text-surface-200/80 transition-colors"
                aria-label="Close modal"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="space-y-2.5">
              <div>
                <span className="text-[10px] uppercase tracking-wider text-surface-200/40 font-semibold">Source Reference</span>
                <p className="text-xs text-surface-200/80 mt-0.5">{sourceRef}</p>
              </div>
              <div>
                <span className="text-[10px] uppercase tracking-wider text-surface-200/40 font-semibold">Metric</span>
                <p className="text-xs text-surface-200/80 mt-0.5">{metricUsed}</p>
              </div>
              {confidenceNote && (
                <div>
                  <span className="text-[10px] uppercase tracking-wider text-surface-200/40 font-semibold">Confidence</span>
                  <p className="text-xs text-surface-200/80 mt-0.5">{confidenceNote}</p>
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
