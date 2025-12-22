'use client';

import React, { forwardRef, ButtonHTMLAttributes, ReactNode } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type ButtonVariant = 'primary' | 'secondary' | 'tertiary' | 'ghost';
export type ButtonSize = 'sm' | 'md' | 'lg' | 'xl';

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
// STYLES
// =============================================================================

const baseStyles = `
  inline-flex items-center justify-center gap-2
  font-heading font-medium
  rounded-sm
  border
  cursor-pointer
  transition-all duration-fast ease-out
  focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy-500 focus-visible:ring-offset-2
  disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none
  select-none
`;

const variantStyles: Record<ButtonVariant, string> = {
  primary: `
    bg-terra-500 text-white border-transparent
    hover:bg-terra-600 hover:-translate-y-0.5 hover:shadow-terra
    active:bg-terra-700 active:translate-y-0
  `,
  secondary: `
    bg-transparent text-navy-500 border-navy-500
    hover:bg-navy-500/5 hover:border-navy-700 hover:text-navy-700
    active:bg-navy-500/10
  `,
  tertiary: `
    bg-transparent text-navy-500 border-transparent
    hover:text-navy-700 hover:underline
    active:text-navy-900
  `,
  ghost: `
    bg-transparent text-neutral-600 border-transparent
    hover:bg-cream-100 hover:text-navy-900
    active:bg-cream-200
  `,
};

const sizeStyles: Record<ButtonSize, string> = {
  sm: 'h-8 px-3 text-body-sm',
  md: 'h-10 px-5 text-body-md',
  lg: 'h-12 px-6 text-body-md',
  xl: 'h-14 px-8 text-body-lg',
};

const iconSizeStyles: Record<ButtonSize, string> = {
  sm: 'w-4 h-4',
  md: 'w-5 h-5',
  lg: 'w-5 h-5',
  xl: 'w-6 h-6',
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
      xl: 'w-14 h-14 p-0',
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
        ${attached ? '[&>button]:rounded-none [&>button:first-child]:rounded-l-sm [&>button:last-child]:rounded-r-sm [&>button:not(:last-child)]:border-r-0' : 'gap-2'}
        ${className}
      `}
      role="group"
    >
      {children}
    </div>
  );
};

export default Button;
