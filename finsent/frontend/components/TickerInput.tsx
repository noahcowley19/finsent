'use client';

import { useState, useRef, KeyboardEvent } from 'react';

interface TickerInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  placeholder?: string;
  disabled?: boolean;
  loading?: boolean;
  className?: string;
}

export default function TickerInput({
  value,
  onChange,
  onSubmit,
  placeholder = 'Enter ticker symbol (e.g., AAPL)',
  disabled = false,
  loading = false,
  className = '',
}: TickerInputProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !disabled && value.trim()) {
      onSubmit();
    }
  };

  return (
    <div className={className} style={{ display: 'flex', gap: '12px' }}>
      <div style={{ position: 'relative', flex: 1 }}>
        <div
          style={{
            position: 'absolute',
            left: '16px',
            top: '50%',
            transform: 'translateY(-50%)',
            color: 'var(--text-muted)',
          }}
        >
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="11" cy="11" r="8" />
            <path d="m21 21-4.3-4.3" />
          </svg>
        </div>
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value.toUpperCase())}
          onKeyPress={handleKeyPress}
          placeholder={placeholder}
          disabled={disabled || loading}
          className="input-field"
          style={{ paddingLeft: '48px' }}
        />
      </div>
      <button
        onClick={onSubmit}
        disabled={disabled || loading || !value.trim()}
        className="btn-primary"
        style={{ minWidth: '120px' }}
      >
        {loading ? (
          <span
            style={{
              display: 'inline-block',
              width: '16px',
              height: '16px',
              border: '2px solid transparent',
              borderTopColor: 'currentColor',
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite',
            }}
          />
        ) : (
          'Analyze'
        )}
      </button>
    </div>
  );
}

interface MultiTickerInputProps {
  tickers: string[];
  onChange: (tickers: string[]) => void;
  onSubmit: () => void;
  maxTickers?: number;
  placeholder?: string;
  disabled?: boolean;
  loading?: boolean;
  className?: string;
}

export function MultiTickerInput({
  tickers,
  onChange,
  onSubmit,
  maxTickers = 10,
  placeholder = 'Add ticker',
  disabled = false,
  loading = false,
  className = '',
}: MultiTickerInputProps) {
  const [input, setInput] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);

  const addTicker = () => {
    const ticker = input.trim().toUpperCase();
    if (ticker && !tickers.includes(ticker) && tickers.length < maxTickers) {
      onChange([...tickers, ticker]);
      setInput('');
      inputRef.current?.focus();
    }
  };

  const removeTicker = (tickerToRemove: string) => {
    onChange(tickers.filter((t) => t !== tickerToRemove));
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addTicker();
    }
  };

  return (
    <div className={className}>
      {/* Ticker chips */}
      {tickers.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginBottom: '16px' }}>
          {tickers.map((ticker) => (
            <div
              key={ticker}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '6px',
                padding: '6px 10px 6px 14px',
                background: 'var(--bg-secondary)',
                border: '1px solid var(--border)',
                borderRadius: '20px',
                fontSize: '13px',
                fontWeight: 600,
                color: 'var(--accent)',
              }}
            >
              {ticker}
              <button
                onClick={() => removeTicker(ticker)}
                style={{
                  background: 'transparent',
                  border: 'none',
                  cursor: 'pointer',
                  color: 'var(--text-muted)',
                  display: 'flex',
                  padding: 0,
                }}
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Input row */}
      <div style={{ display: 'flex', gap: '12px' }}>
        <div style={{ position: 'relative', flex: 1 }}>
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value.toUpperCase())}
            onKeyPress={handleKeyPress}
            placeholder={tickers.length >= maxTickers ? 'Max tickers reached' : placeholder}
            disabled={disabled || loading || tickers.length >= maxTickers}
            className="input-field"
          />
        </div>
        <button
          onClick={addTicker}
          disabled={!input.trim() || tickers.length >= maxTickers}
          className="btn-secondary"
          style={{ minWidth: '80px' }}
        >
          Add
        </button>
        <button
          onClick={onSubmit}
          disabled={disabled || loading || tickers.length === 0}
          className="btn-primary"
          style={{ minWidth: '120px' }}
        >
          {loading ? (
            <span
              style={{
                display: 'inline-block',
                width: '16px',
                height: '16px',
                border: '2px solid transparent',
                borderTopColor: 'currentColor',
                borderRadius: '50%',
                animation: 'spin 0.8s linear infinite',
              }}
            />
          ) : (
            'Analyze'
          )}
        </button>
      </div>

      {/* Counter */}
      <div style={{ marginTop: '8px', fontSize: '12px', color: 'var(--text-muted)' }}>
        {tickers.length} / {maxTickers} tickers
      </div>
    </div>
  );
}
