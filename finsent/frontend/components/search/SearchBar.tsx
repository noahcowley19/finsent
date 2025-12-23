'use client';

// =============================================================================
// SEARCH BAR COMPONENT
// =============================================================================
// Stock search input with autocomplete
//
// Location: frontend/components/search/SearchBar.tsx
//
// =============================================================================

import React, { useState, useRef, useEffect } from 'react';

export interface SearchBarProps {
  /** Initial search value */
  initialValue?: string;
  /** Placeholder text */
  placeholder?: string;
  /** Search handler */
  onSearch: (query: string) => void;
  /** Loading state */
  isLoading?: boolean;
  /** Size variant */
  size?: 'default' | 'large';
  /** Auto focus on mount */
  autoFocus?: boolean;
}

export const SearchBar: React.FC<SearchBarProps> = ({
  initialValue = '',
  placeholder = 'Search by ticker or company name...',
  onSearch,
  isLoading = false,
  size = 'default',
  autoFocus = false,
}) => {
  const [value, setValue] = useState(initialValue);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (autoFocus && inputRef.current) {
      inputRef.current.focus();
    }
  }, [autoFocus]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (value.trim()) {
      onSearch(value.trim());
    }
  };

  const handleClear = () => {
    setValue('');
    inputRef.current?.focus();
  };

  const sizeStyles = {
    default: 'h-12',
    large: 'h-14 lg:h-16',
  };

  const inputSizeStyles = {
    default: 'text-body-md',
    large: 'text-body-lg',
  };

  return (
    <form onSubmit={handleSubmit} className="relative">
      <div className="relative">
        {/* Search icon */}
        <div className="absolute left-4 top-1/2 -translate-y-1/2 text-neutral-400">
          {isLoading ? (
            <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          ) : (
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          )}
        </div>

        {/* Input */}
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder={placeholder}
          className={`
            w-full pl-12 pr-24
            ${sizeStyles[size]}
            ${inputSizeStyles[size]}
            bg-white border border-border-medium rounded-xl
            text-navy-900 placeholder:text-neutral-400
            transition-all duration-fast
            hover:border-border-heavy
            focus:outline-none focus:border-navy-500 focus:ring-2 focus:ring-navy-500/20
          `}
        />

        {/* Clear button */}
        {value && (
          <button
            type="button"
            onClick={handleClear}
            className="absolute right-20 top-1/2 -translate-y-1/2 p-1.5 text-neutral-400 hover:text-neutral-600 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}

        {/* Submit button */}
        <button
          type="submit"
          disabled={!value.trim() || isLoading}
          className={`
            absolute right-2 top-1/2 -translate-y-1/2
            px-4 py-2 rounded-lg
            bg-terra-500 text-white font-medium
            transition-all duration-fast
            hover:bg-terra-600
            disabled:opacity-50 disabled:cursor-not-allowed
          `}
        >
          Search
        </button>
      </div>

      {/* Keyboard hint */}
      <p className="mt-2 text-caption text-neutral-400 text-center">
        Press <kbd className="px-1.5 py-0.5 bg-cream-100 rounded text-neutral-500 font-mono text-[10px]">Enter</kbd> to search
      </p>
    </form>
  );
};

export default SearchBar;
