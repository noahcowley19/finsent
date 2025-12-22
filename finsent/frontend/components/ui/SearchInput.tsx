'use client';

import React, { forwardRef, InputHTMLAttributes, useState, useCallback } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type SearchInputSize = 'sm' | 'md' | 'lg';

export interface SearchInputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size' | 'type'> {
  /** Input size */
  size?: SearchInputSize;
  /** Callback when search is submitted */
  onSearch?: (value: string) => void;
  /** Callback when input is cleared */
  onClear?: () => void;
  /** Show loading spinner */
  isLoading?: boolean;
  /** Full width */
  fullWidth?: boolean;
}

// =============================================================================
// STYLES
// =============================================================================

const sizeStyles: Record<SearchInputSize, string> = {
  sm: 'h-8 pl-9 pr-9 text-body-sm',
  md: 'h-10 pl-11 pr-11 text-body-md',
  lg: 'h-12 pl-12 pr-12 text-body-lg',
};

const iconSizeStyles: Record<SearchInputSize, string> = {
  sm: 'w-4 h-4',
  md: 'w-5 h-5',
  lg: 'w-5 h-5',
};

const iconPositionLeft: Record<SearchInputSize, string> = {
  sm: 'left-3',
  md: 'left-4',
  lg: 'left-4',
};

const iconPositionRight: Record<SearchInputSize, string> = {
  sm: 'right-3',
  md: 'right-4',
  lg: 'right-4',
};

// =============================================================================
// ICONS
// =============================================================================

const SearchIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
    />
  </svg>
);

const ClearIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M6 18L18 6M6 6l12 12"
    />
  </svg>
);

const SpinnerIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg className={`animate-spin ${className}`} fill="none" viewBox="0 0 24 24">
    <circle
      className="opacity-25"
      cx="12"
      cy="12"
      r="10"
      stroke="currentColor"
      strokeWidth="4"
    />
    <path
      className="opacity-75"
      fill="currentColor"
      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
    />
  </svg>
);

// =============================================================================
// COMPONENT
// =============================================================================

export const SearchInput = forwardRef<HTMLInputElement, SearchInputProps>(
  (
    {
      size = 'md',
      onSearch,
      onClear,
      isLoading = false,
      fullWidth = true,
      disabled,
      value,
      onChange,
      onKeyDown,
      className = '',
      placeholder = 'Search...',
      ...props
    },
    ref
  ) => {
    const [internalValue, setInternalValue] = useState('');
    const isControlled = value !== undefined;
    const currentValue = isControlled ? value : internalValue;
    const hasValue = Boolean(currentValue);

    const handleChange = useCallback(
      (e: React.ChangeEvent<HTMLInputElement>) => {
        if (!isControlled) {
          setInternalValue(e.target.value);
        }
        onChange?.(e);
      },
      [isControlled, onChange]
    );

    const handleClear = useCallback(() => {
      if (!isControlled) {
        setInternalValue('');
      }
      onClear?.();
      // Create a synthetic event for onChange
      const syntheticEvent = {
        target: { value: '' },
      } as React.ChangeEvent<HTMLInputElement>;
      onChange?.(syntheticEvent);
    }, [isControlled, onClear, onChange]);

    const handleKeyDown = useCallback(
      (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' && onSearch) {
          onSearch(String(currentValue));
        }
        if (e.key === 'Escape' && hasValue) {
          handleClear();
        }
        onKeyDown?.(e);
      },
      [currentValue, onSearch, hasValue, handleClear, onKeyDown]
    );

    return (
      <div className={`relative ${fullWidth ? 'w-full' : 'inline-block'}`}>
        {/* Search icon */}
        <span
          className={`
            absolute top-1/2 -translate-y-1/2 ${iconPositionLeft[size]}
            ${iconSizeStyles[size]}
            text-neutral-400
            pointer-events-none
          `}
        >
          <SearchIcon className={iconSizeStyles[size]} />
        </span>

        {/* Input */}
        <input
          ref={ref}
          type="search"
          value={currentValue}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          disabled={disabled || isLoading}
          placeholder={placeholder}
          className={`
            w-full
            font-body
            text-navy-900
            bg-white
            border border-border-medium
            rounded-sm
            transition-all duration-fast ease-out
            placeholder:text-neutral-400
            hover:border-border-heavy
            focus:outline-none focus:border-navy-500 focus:ring-2 focus:ring-navy-500/10
            disabled:bg-cream-100 disabled:text-neutral-400 disabled:cursor-not-allowed
            [&::-webkit-search-cancel-button]:hidden
            [&::-webkit-search-decoration]:hidden
            ${sizeStyles[size]}
            ${className}
          `.trim().replace(/\s+/g, ' ')}
          {...props}
        />

        {/* Clear/Loading button */}
        {(hasValue || isLoading) && (
          <button
            type="button"
            onClick={isLoading ? undefined : handleClear}
            disabled={disabled || isLoading}
            className={`
              absolute top-1/2 -translate-y-1/2 ${iconPositionRight[size]}
              ${iconSizeStyles[size]}
              text-neutral-400
              transition-colors duration-fast
              ${isLoading ? 'cursor-default' : 'hover:text-navy-500 cursor-pointer'}
              disabled:opacity-50
            `}
            aria-label="Clear search"
          >
            {isLoading ? (
              <SpinnerIcon className={iconSizeStyles[size]} />
            ) : (
              <ClearIcon className={iconSizeStyles[size]} />
            )}
          </button>
        )}
      </div>
    );
  }
);

SearchInput.displayName = 'SearchInput';

export default SearchInput;
