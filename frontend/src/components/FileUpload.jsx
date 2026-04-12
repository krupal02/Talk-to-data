/**
 * FileUpload – drag-and-drop or click-to-browse file upload component.
 *
 * Accepts CSV and SQLite files, shows upload progress, and displays
 * a preview table of the first five rows after a successful upload.
 */

import { useState, useRef, useCallback } from 'react';
import { uploadFile } from '../api/client';

const ACCEPTED = '.csv,.db,.sqlite';

export default function FileUpload({ onUploadSuccess }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFile = useCallback(async (file) => {
    if (!file) return;
    setUploading(true);
    setError(null);
    setUploadResult(null);

    try {
      const data = await uploadFile(file);
      setUploadResult(data);
      onUploadSuccess(data);
    } catch (err) {
      const message =
        err.response?.data?.detail ||
        err.message ||
        'Upload failed. Please try again.';
      setError(message);
    } finally {
      setUploading(false);
    }
  }, [onUploadSuccess]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  const onDragLeave = useCallback(() => {
    setDragging(false);
  }, []);

  const onInputChange = useCallback((e) => {
    const file = e.target.files[0];
    handleFile(file);
  }, [handleFile]);

  return (
    <div className="w-full animate-fade-in">
      {/* Drop zone */}
      <div
        id="file-drop-zone"
        className={`drop-zone rounded-2xl p-10 text-center cursor-pointer transition-all duration-300 ${
          dragging ? 'dragging' : ''
        } ${uploading ? 'opacity-60 pointer-events-none' : ''}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => fileInputRef.current?.click()}
        role="button"
        tabIndex={0}
        aria-label="Upload a dataset file"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED}
          onChange={onInputChange}
          className="hidden"
          id="file-input"
        />

        {uploading ? (
          <div className="flex flex-col items-center gap-4">
            <div className="w-10 h-10 border-3 border-primary-400 border-t-transparent rounded-full animate-spin" />
            <p className="text-surface-200 font-medium">Processing your file…</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4">
            <div className="w-16 h-16 rounded-2xl gradient-primary flex items-center justify-center shadow-lg shadow-primary-500/20">
              <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
              </svg>
            </div>
            <div>
              <p className="text-lg font-semibold text-surface-100">
                Drop your dataset here
              </p>
              <p className="text-sm text-surface-200/60 mt-1">
                or click to browse · CSV, SQLite supported · 50 MB max
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Error message */}
      {error && (
        <div id="upload-error" className="mt-4 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-300 text-sm animate-slide-up">
          {error}
        </div>
      )}

      {/* Upload result preview */}
      {uploadResult && (
        <div id="upload-preview" className="mt-6 glass rounded-2xl p-6 animate-slide-up">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                <svg className="w-4 h-4 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <span className="font-semibold text-surface-100">Dataset loaded</span>
            </div>
            <span className="text-xs text-surface-200/60 font-mono">
              {uploadResult.row_count.toLocaleString()} rows · {uploadResult.columns.length} columns
            </span>
          </div>

          {/* Column badges */}
          <div className="flex flex-wrap gap-2 mb-4">
            {uploadResult.columns.map((col) => (
              <span
                key={col}
                className="px-2.5 py-1 rounded-lg bg-primary-500/10 text-primary-300 text-xs font-medium border border-primary-500/20"
              >
                {col}
              </span>
            ))}
          </div>

          {/* Preview table */}
          {uploadResult.preview.length > 0 && (
            <div className="overflow-x-auto rounded-xl border border-surface-200/10">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-surface-800/50">
                    {uploadResult.columns.map((col) => (
                      <th key={col} className="px-3 py-2 text-left font-medium text-surface-200/80 whitespace-nowrap">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {uploadResult.preview.map((row, i) => (
                    <tr key={i} className="border-t border-surface-200/5 hover:bg-surface-700/20 transition-colors">
                      {uploadResult.columns.map((col) => (
                        <td key={col} className="px-3 py-2 text-surface-200/70 whitespace-nowrap">
                          {String(row[col] ?? '')}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
