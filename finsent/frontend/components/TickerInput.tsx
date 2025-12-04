'use client';

import { useState, FormEvent } from 'react';
import { cn } from '@/lib/utils';
import LoadingSpinner from './LoadingSpinner';

interface TickerInputProps {
  onSubmit: (ticker: string) => void;
  loading?: boolean;
  placeholder?: string;
  buttonText?: string;
  className?: string;
}

export default function TickerInput({
  onSubmit,
  loading = false,
  placeholder = 'Enter ticker symbol (e.g., AAPL)',
  buttonText = 'Analyze',
  className
}: TickerInputProps) {
  const [ticker, setTicker] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = ticker.trim().toUpperCase();
    if (trimmed) {
      onSubmit(trimmed);
    }
  };

  return (
    <form onSubmit={handleSubmit} className={cn('flex gap-3', className)}>
      <input
        type="text"
        value={ticker}
        onChange={(e) => setTicker(e.target.value.toUpperCase())}
        placeholder={placeholder}
        className="input-field flex-1 uppercase"
        disabled={loading}
      />
      <button
        type="submit"
        disabled={loading || !ticker.trim()}
        className="btn-primary min-w-[120px] flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <LoadingSpinner size="sm" className="border-white border-t-transparent" />
            <span>Loading...</span>
          </>
        ) : (
          buttonText
        )}
      </button>
    </form>
  );
}

interface MultiTickerInputProps {
  onSubmit: (tickers: string[]) => void;
  loading?: boolean;
  placeholder?: string;
  buttonText?: string;
  className?: string;
}

export function MultiTickerInput({
  onSubmit,
  loading = false,
  placeholder = 'Enter tickers separated by commas (e.g., AAPL, MSFT, GOOGL)',
  buttonText = 'Screen',
  className
}: MultiTickerInputProps) {
  const [input, setInput] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const tickers = input
      .split(',')
      .map(t => t.trim().toUpperCase())
      .filter(t => t.length > 0);
    
    if (tickers.length > 0) {
      onSubmit(tickers);
    }
  };

  return (
    <form onSubmit={handleSubmit} className={cn('flex gap-3', className)}>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value.toUpperCase())}
        placeholder={placeholder}
        className="input-field flex-1 uppercase"
        disabled={loading}
      />
      <button
        type="submit"
        disabled={loading || !input.trim()}
        className="btn-primary min-w-[120px] flex items-center justify-center gap-2"
      >
        {loading ? (
          <>
            <LoadingSpinner size="sm" className="border-white border-t-transparent" />
            <span>Loading...</span>
          </>
        ) : (
          buttonText
        )}
      </button>
    </form>
  );
}
