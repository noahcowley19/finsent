'use client';

import React, { forwardRef, ButtonHTMLAttributes, ReactNode } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'accent';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Visual style variant */
  variant?: ButtonVariant;
  /** Button size */
  size?: ButtonSize;
  /** Show loading spinner */
  isLoading?: boolean;
  /** Icon to show before text */
  leftIcon?: ReactNode;
  /** Icon to show after text */
  rightIcon?: ReactNode;
  /** Make button full width */
  fullWidth?: boolean;
  /** Content */
  children: ReactNode;
}

// =============================================================================
// STYLES - Modern SaaS Aesthetic
// =============================================================================

const baseStyles = `
  inline-flex items-center justify-center gap-2
  font-medium tracking-tight
  rounded-lg
  border
  cursor-pointer
  transition-all duration-150 ease-out
  focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent focus-visible:ring-offset-2
  disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none
  select-none
`;

const variantStyles: Record<ButtonVariant, string> = {
  primary: `
    bg-ink-900 text-white border-transparent
    hover:bg-ink-800 hover:-translate-y-px hover:shadow-lg
    active:bg-ink-950 active:translate-y-0
  `,
  secondary: `
    bg-white text-ink-700 border-ink-200
    hover:bg-ink-50 hover:border-ink-300 hover:-translate-y-px
    active:bg-ink-100 active:translate-y-0
  `,
  ghost: `
    bg-transparent text-ink-600 border-transparent
    hover:bg-ink-100 hover:text-ink-900
    active:bg-ink-200
  `,
  accent: `
    bg-accent text-white border-transparent
    hover:bg-accent-600 hover:-translate-y-px hover:shadow-glow-sm
    active:bg-accent-700 active:translate-y-0
  `,
};

const sizeStyles: Record<ButtonSize, string> = {
  sm: 'h-8 px-3 text-body-sm',
  md: 'h-10 px-4 text-body-sm',
  lg: 'h-12 px-6 text-body-md',
};

const iconSizeStyles: Record<ButtonSize, string> = {
  sm: 'w-4 h-4',
  md: 'w-4 h-4',
  lg: 'w-5 h-5',
};

// =============================================================================
// SPINNER COMPONENT
// =============================================================================

const ButtonSpinner: React.FC<{ size: ButtonSize }> = ({ size }) => (
  <svg
    className={`animate-spin ${iconSizeStyles[size]}`}
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
  >
    <circle
      className="opacity-25"
      cx="12"
      cy="12"
      r="10"
      stroke="currentColor"
      strokeWidth="3"
    />
    <path
      className="opacity-75"
      fill="currentColor"
      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
    />
  </svg>
);

// =============================================================================
// BUTTON COMPONENT
// =============================================================================

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      variant = 'primary',
      size = 'md',
      isLoading = false,
      leftIcon,
      rightIcon,
      fullWidth = false,
      disabled,
      children,
      className = '',
      ...props
    },
    ref
  ) => {
    const isDisabled = disabled || isLoading;

    return (
      <button
        ref={ref}
        disabled={isDisabled}
        className={`
          ${baseStyles}
          ${variantStyles[variant]}
          ${sizeStyles[size]}
          ${fullWidth ? 'w-full' : ''}
          ${className}
        `.trim().replace(/\s+/g, ' ')}
        {...props}
      >
        {isLoading ? (
          <ButtonSpinner size={size} />
        ) : leftIcon ? (
          <span className={iconSizeStyles[size]}>{leftIcon}</span>
        ) : null}

        <span className={isLoading ? 'opacity-0' : ''}>{children}</span>

        {!isLoading && rightIcon && (
          <span className={iconSizeStyles[size]}>{rightIcon}</span>
        )}
      </button>
    );
  }
);

Button.displayName = 'Button';

// =============================================================================
// ICON BUTTON VARIANT
// =============================================================================

export interface IconButtonProps extends Omit<ButtonProps, 'children' | 'leftIcon' | 'rightIcon'> {
  /** Icon to display */
  icon: ReactNode;
  /** Accessible label */
  'aria-label': string;
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ icon, size = 'md', className = '', ...props }, ref) => {
    const iconOnlySizes: Record<ButtonSize, string> = {
      sm: 'w-8 h-8 p-0',
      md: 'w-10 h-10 p-0',
      lg: 'w-12 h-12 p-0',
    };

    return (
      <Button
        ref={ref}
        size={size}
        className={`${iconOnlySizes[size]} ${className}`}
        {...props}
      >
        <span className={iconSizeStyles[size]}>{icon}</span>
      </Button>
    );
  }
);

IconButton.displayName = 'IconButton';

// =============================================================================
// BUTTON GROUP
// =============================================================================

export interface ButtonGroupProps {
  children: ReactNode;
  /** Attach buttons together */
  attached?: boolean;
  className?: string;
}

export const ButtonGroup: React.FC<ButtonGroupProps> = ({
  children,
  attached = false,
  className = '',
}) => {
  return (
    <div
      className={`
        inline-flex
        ${attached ? '[&>button]:rounded-none [&>button:first-child]:rounded-l-lg [&>button:last-child]:rounded-r-lg [&>button:not(:last-child)]:border-r-0' : 'gap-2'}
        ${className}
      `}
      role="group"
    >
      {children}
    </div>
  );
};

export default Button;
