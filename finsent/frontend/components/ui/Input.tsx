'use client';

import React, { forwardRef, InputHTMLAttributes, ReactNode, useState } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type InputSize = 'sm' | 'md' | 'lg';

export interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  /** Label text */
  label?: string;
  /** Error message */
  error?: string;
  /** Helper/hint text */
  hint?: string;
  /** Icon on the left side */
  leftIcon?: ReactNode;
  /** Icon on the right side */
  rightIcon?: ReactNode;
  /** Input size */
  size?: InputSize;
  /** Full width */
  fullWidth?: boolean;
}

// =============================================================================
// STYLES
// =============================================================================

const baseInputStyles = `
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
`;

const sizeStyles: Record<InputSize, string> = {
  sm: 'h-8 px-3 text-body-sm',
  md: 'h-10 px-4 text-body-md',
  lg: 'h-12 px-4 text-body-lg',
};

const iconPaddingLeft: Record<InputSize, string> = {
  sm: 'pl-9',
  md: 'pl-11',
  lg: 'pl-12',
};

const iconPaddingRight: Record<InputSize, string> = {
  sm: 'pr-9',
  md: 'pr-11',
  lg: 'pr-12',
};

const iconSizeStyles: Record<InputSize, string> = {
  sm: 'w-4 h-4',
  md: 'w-5 h-5',
  lg: 'w-5 h-5',
};

const iconPositionStyles: Record<InputSize, { left: string; right: string }> = {
  sm: { left: 'left-3', right: 'right-3' },
  md: { left: 'left-4', right: 'right-4' },
  lg: { left: 'left-4', right: 'right-4' },
};

// =============================================================================
// COMPONENT
// =============================================================================

export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      label,
      error,
      hint,
      leftIcon,
      rightIcon,
      size = 'md',
      fullWidth = true,
      disabled,
      className = '',
      id,
      ...props
    },
    ref
  ) => {
    const inputId = id || `input-${Math.random().toString(36).substr(2, 9)}`;
    const hasError = Boolean(error);

    return (
      <div className={`${fullWidth ? 'w-full' : 'inline-block'}`}>
        {/* Label */}
        {label && (
          <label
            htmlFor={inputId}
            className="block text-body-sm font-medium text-navy-700 mb-2"
          >
            {label}
          </label>
        )}

        {/* Input wrapper */}
        <div className="relative">
          {/* Left icon */}
          {leftIcon && (
            <span
              className={`
                absolute top-1/2 -translate-y-1/2 ${iconPositionStyles[size].left}
                ${iconSizeStyles[size]}
                text-neutral-400
                pointer-events-none
              `}
            >
              {leftIcon}
            </span>
          )}

          {/* Input */}
          <input
            ref={ref}
            id={inputId}
            disabled={disabled}
            aria-invalid={hasError}
            aria-describedby={
              hasError ? `${inputId}-error` : hint ? `${inputId}-hint` : undefined
            }
            className={`
              ${baseInputStyles}
              ${sizeStyles[size]}
              ${leftIcon ? iconPaddingLeft[size] : ''}
              ${rightIcon ? iconPaddingRight[size] : ''}
              ${hasError ? 'border-error-500 focus:border-error-500 focus:ring-error-500/10' : ''}
              ${className}
            `.trim().replace(/\s+/g, ' ')}
            {...props}
          />

          {/* Right icon */}
          {rightIcon && (
            <span
              className={`
                absolute top-1/2 -translate-y-1/2 ${iconPositionStyles[size].right}
                ${iconSizeStyles[size]}
                text-neutral-400
                pointer-events-none
              `}
            >
              {rightIcon}
            </span>
          )}
        </div>

        {/* Error message */}
        {hasError && (
          <p
            id={`${inputId}-error`}
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
            id={`${inputId}-hint`}
            className="mt-1.5 text-body-sm text-neutral-500"
          >
            {hint}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

// =============================================================================
// PASSWORD INPUT
// =============================================================================

export interface PasswordInputProps extends Omit<InputProps, 'type' | 'rightIcon'> {
  /** Show password strength indicator */
  showStrength?: boolean;
}

export const PasswordInput = forwardRef<HTMLInputElement, PasswordInputProps>(
  ({ showStrength = false, ...props }, ref) => {
    const [showPassword, setShowPassword] = useState(false);
    const [strength, setStrength] = useState(0);

    const calculateStrength = (password: string): number => {
      let score = 0;
      if (password.length >= 8) score++;
      if (password.length >= 12) score++;
      if (/[a-z]/.test(password) && /[A-Z]/.test(password)) score++;
      if (/\d/.test(password)) score++;
      if (/[^a-zA-Z0-9]/.test(password)) score++;
      return Math.min(score, 4);
    };

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      if (showStrength) {
        setStrength(calculateStrength(e.target.value));
      }
      props.onChange?.(e);
    };

    const strengthColors = ['bg-neutral-200', 'bg-error-500', 'bg-warning-500', 'bg-success-400', 'bg-success-500'];
    const strengthLabels = ['', 'Weak', 'Fair', 'Good', 'Strong'];

    const toggleButton = (
      <button
        type="button"
        onClick={() => setShowPassword(!showPassword)}
        className="absolute right-4 top-1/2 -translate-y-1/2 text-neutral-400 hover:text-navy-500 transition-colors"
        aria-label={showPassword ? 'Hide password' : 'Show password'}
      >
        {showPassword ? (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
          </svg>
        ) : (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
          </svg>
        )}
      </button>
    );

    return (
      <div className="w-full">
        <div className="relative">
          <Input
            ref={ref}
            type={showPassword ? 'text' : 'password'}
            {...props}
            onChange={handleChange}
            className="pr-12"
          />
          {toggleButton}
        </div>
        
        {showStrength && (
          <div className="mt-2">
            <div className="flex gap-1 mb-1">
              {[1, 2, 3, 4].map((level) => (
                <div
                  key={level}
                  className={`h-1 flex-1 rounded-full transition-colors ${
                    level <= strength ? strengthColors[strength] : 'bg-neutral-200'
                  }`}
                />
              ))}
            </div>
            {strength > 0 && (
              <p className="text-caption text-neutral-500">
                Password strength: {strengthLabels[strength]}
              </p>
            )}
          </div>
        )}
      </div>
    );
  }
);

PasswordInput.displayName = 'PasswordInput';

export default Input;
