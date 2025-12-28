'use client';

import React, { forwardRef, InputHTMLAttributes, ReactNode } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type InputSize = 'sm' | 'md' | 'lg';
export type InputVariant = 'default' | 'ghost';

export interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  /** Input size */
  size?: InputSize;
  /** Visual variant */
  variant?: InputVariant;
  /** Error state */
  error?: boolean;
  /** Error message */
  errorMessage?: string;
  /** Helper text */
  helperText?: string;
  /** Label */
  label?: string;
  /** Left icon/element */
  leftElement?: ReactNode;
  /** Right icon/element */
  rightElement?: ReactNode;
  /** Full width */
  fullWidth?: boolean;
}

// =============================================================================
// STYLES - Modern SaaS Aesthetic
// =============================================================================

const baseInputStyles = `
  w-full
  font-normal tracking-tight
  text-ink-900
  bg-white
  border
  rounded-lg
  transition-all duration-150 ease-out
  placeholder:text-ink-400
  focus:outline-none
  disabled:bg-ink-50 disabled:text-ink-400 disabled:cursor-not-allowed
`;

const sizeStyles: Record<InputSize, string> = {
  sm: 'h-8 px-3 text-body-sm',
  md: 'h-10 px-3.5 text-body-sm',
  lg: 'h-12 px-4 text-body-md',
};

const variantStyles: Record<InputVariant, { default: string; focus: string; error: string }> = {
  default: {
    default: 'border-ink-200 hover:border-ink-300',
    focus: 'focus:border-accent focus:ring-2 focus:ring-accent/10',
    error: 'border-error-500 focus:border-error-500 focus:ring-error-500/10',
  },
  ghost: {
    default: 'border-transparent bg-ink-50 hover:bg-ink-100',
    focus: 'focus:bg-white focus:border-ink-200 focus:ring-2 focus:ring-ink-200/50',
    error: 'border-error-500 bg-error-50 focus:ring-error-500/10',
  },
};

// =============================================================================
// INPUT COMPONENT
// =============================================================================

export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      size = 'md',
      variant = 'default',
      error = false,
      errorMessage,
      helperText,
      label,
      leftElement,
      rightElement,
      fullWidth = true,
      className = '',
      id,
      ...props
    },
    ref
  ) => {
    const inputId = id || `input-${Math.random().toString(36).substr(2, 9)}`;
    const styles = variantStyles[variant];

    const inputClasses = `
      ${baseInputStyles}
      ${sizeStyles[size]}
      ${error ? styles.error : `${styles.default} ${styles.focus}`}
      ${leftElement ? 'pl-10' : ''}
      ${rightElement ? 'pr-10' : ''}
      ${className}
    `.trim().replace(/\s+/g, ' ');

    return (
      <div className={`${fullWidth ? 'w-full' : 'inline-block'}`}>
        {label && (
          <label
            htmlFor={inputId}
            className="block text-body-sm font-medium text-ink-700 mb-1.5"
          >
            {label}
          </label>
        )}

        <div className="relative">
          {leftElement && (
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-ink-400">
              {leftElement}
            </div>
          )}

          <input
            ref={ref}
            id={inputId}
            className={inputClasses}
            aria-invalid={error}
            aria-describedby={
              error && errorMessage
                ? `${inputId}-error`
                : helperText
                  ? `${inputId}-helper`
                  : undefined
            }
            {...props}
          />

          {rightElement && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2 text-ink-400">
              {rightElement}
            </div>
          )}
        </div>

        {error && errorMessage && (
          <p id={`${inputId}-error`} className="mt-1.5 text-body-xs text-error-600">
            {errorMessage}
          </p>
        )}

        {!error && helperText && (
          <p id={`${inputId}-helper`} className="mt-1.5 text-body-xs text-ink-500">
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

// =============================================================================
// SEARCH INPUT
// =============================================================================

export interface SearchInputProps extends Omit<InputProps, 'leftElement' | 'type'> {
  /** Show loading state */
  isLoading?: boolean;
  /** Show clear button */
  showClear?: boolean;
  /** Clear handler */
  onClear?: () => void;
}

export const SearchInput = forwardRef<HTMLInputElement, SearchInputProps>(
  ({ isLoading = false, showClear = false, onClear, value, ...props }, ref) => {
    const showClearButton = showClear && value && String(value).length > 0;

    return (
      <Input
        ref={ref}
        type="search"
        value={value}
        leftElement={
          isLoading ? (
            <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
          ) : (
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          )
        }
        rightElement={
          showClearButton ? (
            <button
              type="button"
              onClick={onClear}
              className="p-0.5 rounded hover:bg-ink-100 transition-colors"
              aria-label="Clear search"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          ) : null
        }
        {...props}
      />
    );
  }
);

SearchInput.displayName = 'SearchInput';

export default Input;
