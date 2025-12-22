'use client';

import React, { forwardRef, TextareaHTMLAttributes, useRef, useEffect } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  /** Label text */
  label?: string;
  /** Error message */
  error?: string;
  /** Helper/hint text */
  hint?: string;
  /** Auto-resize based on content */
  autoResize?: boolean;
  /** Maximum height when auto-resizing */
  maxHeight?: number;
  /** Show character count */
  showCount?: boolean;
}

// =============================================================================
// COMPONENT
// =============================================================================

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  (
    {
      label,
      error,
      hint,
      autoResize = false,
      maxHeight = 300,
      showCount = false,
      maxLength,
      disabled,
      className = '',
      id,
      value,
      onChange,
      ...props
    },
    ref
  ) => {
    const textareaId = id || `textarea-${Math.random().toString(36).substr(2, 9)}`;
    const hasError = Boolean(error);
    const internalRef = useRef<HTMLTextAreaElement>(null);
    const textareaRef = (ref as React.RefObject<HTMLTextAreaElement>) || internalRef;

    // Auto-resize logic
    useEffect(() => {
      if (autoResize && textareaRef.current) {
        const textarea = textareaRef.current;
        textarea.style.height = 'auto';
        const newHeight = Math.min(textarea.scrollHeight, maxHeight);
        textarea.style.height = `${newHeight}px`;
      }
    }, [value, autoResize, maxHeight, textareaRef]);

    const currentLength = typeof value === 'string' ? value.length : 0;

    return (
      <div className="w-full">
        {/* Label */}
        {label && (
          <label
            htmlFor={textareaId}
            className="block text-body-sm font-medium text-navy-700 mb-2"
          >
            {label}
          </label>
        )}

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          id={textareaId}
          disabled={disabled}
          value={value}
          onChange={onChange}
          maxLength={maxLength}
          aria-invalid={hasError}
          aria-describedby={
            hasError ? `${textareaId}-error` : hint ? `${textareaId}-hint` : undefined
          }
          className={`
            w-full
            min-h-[100px]
            px-4 py-3
            font-body text-body-md text-navy-900
            bg-white
            border border-border-medium
            rounded-sm
            resize-y
            transition-all duration-fast ease-out
            placeholder:text-neutral-400
            hover:border-border-heavy
            focus:outline-none focus:border-navy-500 focus:ring-2 focus:ring-navy-500/10
            disabled:bg-cream-100 disabled:text-neutral-400 disabled:cursor-not-allowed disabled:resize-none
            ${hasError ? 'border-error-500 focus:border-error-500 focus:ring-error-500/10' : ''}
            ${autoResize ? 'resize-none overflow-hidden' : ''}
            ${className}
          `.trim().replace(/\s+/g, ' ')}
          {...props}
        />

        {/* Footer: Error/Hint + Character count */}
        <div className="flex justify-between items-start mt-1.5 gap-4">
          <div className="flex-1">
            {/* Error message */}
            {hasError && (
              <p
                id={`${textareaId}-error`}
                className="text-body-sm text-error-500 flex items-center gap-1"
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
                id={`${textareaId}-hint`}
                className="text-body-sm text-neutral-500"
              >
                {hint}
              </p>
            )}
          </div>

          {/* Character count */}
          {showCount && maxLength && (
            <p
              className={`text-caption flex-shrink-0 ${
                currentLength >= maxLength ? 'text-error-500' : 'text-neutral-400'
              }`}
            >
              {currentLength}/{maxLength}
            </p>
          )}
        </div>
      </div>
    );
  }
);

Textarea.displayName = 'Textarea';

export default Textarea;
