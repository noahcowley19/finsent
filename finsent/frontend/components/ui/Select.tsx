'use client';

import React, { forwardRef, SelectHTMLAttributes, ReactNode } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type SelectSize = 'sm' | 'md' | 'lg';

export interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

export interface SelectProps extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'size'> {
  /** Label text */
  label?: string;
  /** Error message */
  error?: string;
  /** Helper/hint text */
  hint?: string;
  /** Select size */
  size?: SelectSize;
  /** Options array */
  options?: SelectOption[];
  /** Placeholder text */
  placeholder?: string;
  /** Left icon */
  leftIcon?: ReactNode;
  /** Full width */
  fullWidth?: boolean;
}

// =============================================================================
// STYLES
// =============================================================================

const sizeStyles: Record<SelectSize, string> = {
  sm: 'h-8 px-3 pr-10 text-body-sm',
  md: 'h-10 px-4 pr-11 text-body-md',
  lg: 'h-12 px-4 pr-12 text-body-lg',
};

const iconPaddingLeft: Record<SelectSize, string> = {
  sm: 'pl-9',
  md: 'pl-11',
  lg: 'pl-12',
};

const iconSizeStyles: Record<SelectSize, string> = {
  sm: 'w-4 h-4',
  md: 'w-5 h-5',
  lg: 'w-5 h-5',
};

// =============================================================================
// COMPONENT
// =============================================================================

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  (
    {
      label,
      error,
      hint,
      size = 'md',
      options = [],
      placeholder,
      leftIcon,
      fullWidth = true,
      disabled,
      className = '',
      id,
      children,
      ...props
    },
    ref
  ) => {
    const selectId = id || `select-${Math.random().toString(36).substr(2, 9)}`;
    const hasError = Boolean(error);

    return (
      <div className={`${fullWidth ? 'w-full' : 'inline-block'}`}>
        {/* Label */}
        {label && (
          <label
            htmlFor={selectId}
            className="block text-body-sm font-medium text-navy-700 mb-2"
          >
            {label}
          </label>
        )}

        {/* Select wrapper */}
        <div className="relative">
          {/* Left icon */}
          {leftIcon && (
            <span
              className={`
                absolute top-1/2 -translate-y-1/2 left-4
                ${iconSizeStyles[size]}
                text-neutral-400
                pointer-events-none
              `}
            >
              {leftIcon}
            </span>
          )}

          {/* Select */}
          <select
            ref={ref}
            id={selectId}
            disabled={disabled}
            aria-invalid={hasError}
            aria-describedby={
              hasError ? `${selectId}-error` : hint ? `${selectId}-hint` : undefined
            }
            className={`
              w-full
              font-body
              text-navy-900
              bg-white
              border border-border-medium
              rounded-sm
              cursor-pointer
              appearance-none
              transition-all duration-fast ease-out
              hover:border-border-heavy
              focus:outline-none focus:border-navy-500 focus:ring-2 focus:ring-navy-500/10
              disabled:bg-cream-100 disabled:text-neutral-400 disabled:cursor-not-allowed
              ${sizeStyles[size]}
              ${leftIcon ? iconPaddingLeft[size] : ''}
              ${hasError ? 'border-error-500 focus:border-error-500 focus:ring-error-500/10' : ''}
              ${className}
            `.trim().replace(/\s+/g, ' ')}
            {...props}
          >
            {placeholder && (
              <option value="" disabled>
                {placeholder}
              </option>
            )}
            {children ||
              options.map((option) => (
                <option
                  key={option.value}
                  value={option.value}
                  disabled={option.disabled}
                >
                  {option.label}
                </option>
              ))}
          </select>

          {/* Dropdown arrow */}
          <span className="absolute top-1/2 -translate-y-1/2 right-4 pointer-events-none text-neutral-400">
            <svg
              className={iconSizeStyles[size]}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 9l-7 7-7-7"
              />
            </svg>
          </span>
        </div>

        {/* Error message */}
        {hasError && (
          <p
            id={`${selectId}-error`}
            className="mt-1.5 text-body-sm text-error-500 flex items-center gap-1"
          >
            <svg
              className="w-4 h-4 flex-shrink-0"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
            {error}
          </p>
        )}

        {/* Hint text */}
        {hint && !hasError && (
          <p
            id={`${selectId}-hint`}
            className="mt-1.5 text-body-sm text-neutral-500"
          >
            {hint}
          </p>
        )}
      </div>
    );
  }
);

Select.displayName = 'Select';

export default Select;
