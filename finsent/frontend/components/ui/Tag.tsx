'use client';

import React, { HTMLAttributes, ReactNode, KeyboardEvent } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type TagVariant = 'default' | 'primary' | 'success' | 'warning' | 'error';
export type TagSize = 'sm' | 'md';

export interface TagProps extends HTMLAttributes<HTMLSpanElement> {
  /** Color variant */
  variant?: TagVariant;
  /** Tag size */
  size?: TagSize;
  /** Make tag removable */
  removable?: boolean;
  /** Called when remove button is clicked */
  onRemove?: () => void;
  /** Make tag clickable/selectable */
  clickable?: boolean;
  /** Selected state for clickable tags */
  selected?: boolean;
  /** Icon to show before text */
  icon?: ReactNode;
  /** Content */
  children: ReactNode;
}

// =============================================================================
// STYLES
// =============================================================================

const baseStyles = `
  inline-flex items-center gap-1.5
  font-medium
  rounded-md
  transition-all duration-fast ease-out
`;

const variantStyles: Record<TagVariant, string> = {
  default: 'bg-cream-100 text-navy-700 border border-border-light',
  primary: 'bg-navy-50 text-navy-700 border border-navy-200',
  success: 'bg-success-50 text-success-700 border border-success-200',
  warning: 'bg-warning-50 text-warning-700 border border-warning-200',
  error: 'bg-error-50 text-error-700 border border-error-200',
};

const selectedStyles: Record<TagVariant, string> = {
  default: 'bg-navy-100 text-navy-900 border-navy-300',
  primary: 'bg-navy-500 text-white border-navy-500',
  success: 'bg-success-500 text-white border-success-500',
  warning: 'bg-warning-500 text-white border-warning-500',
  error: 'bg-error-500 text-white border-error-500',
};

const hoverStyles: Record<TagVariant, string> = {
  default: 'hover:bg-cream-200 hover:border-border-medium',
  primary: 'hover:bg-navy-100 hover:border-navy-300',
  success: 'hover:bg-success-100 hover:border-success-300',
  warning: 'hover:bg-warning-100 hover:border-warning-300',
  error: 'hover:bg-error-100 hover:border-error-300',
};

const sizeStyles: Record<TagSize, string> = {
  sm: 'px-2 py-0.5 text-caption',
  md: 'px-3 py-1 text-body-sm',
};

// =============================================================================
// COMPONENT
// =============================================================================

export const Tag: React.FC<TagProps> = ({
  variant = 'default',
  size = 'md',
  removable = false,
  onRemove,
  clickable = false,
  selected = false,
  icon,
  children,
  className = '',
  onClick,
  ...props
}) => {
  const isInteractive = clickable || removable || onClick;

  const handleKeyDown = (e: KeyboardEvent<HTMLSpanElement>) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      if (onClick) {
        onClick(e as unknown as React.MouseEvent<HTMLSpanElement>);
      }
    }
    if (e.key === 'Backspace' || e.key === 'Delete') {
      if (removable && onRemove) {
        e.preventDefault();
        onRemove();
      }
    }
  };

  const handleRemoveClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    onRemove?.();
  };

  return (
    <span
      role={clickable ? 'button' : undefined}
      tabIndex={isInteractive ? 0 : undefined}
      onClick={onClick}
      onKeyDown={isInteractive ? handleKeyDown : undefined}
      className={`
        ${baseStyles}
        ${selected ? selectedStyles[variant] : variantStyles[variant]}
        ${sizeStyles[size]}
        ${clickable ? `cursor-pointer ${hoverStyles[variant]}` : ''}
        ${clickable ? 'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy-500 focus-visible:ring-offset-1' : ''}
        ${className}
      `.trim().replace(/\s+/g, ' ')}
      {...props}
    >
      {icon && (
        <span className={size === 'sm' ? 'w-3 h-3' : 'w-4 h-4'}>
          {icon}
        </span>
      )}
      
      <span>{children}</span>

      {removable && (
        <button
          type="button"
          onClick={handleRemoveClick}
          className={`
            ${size === 'sm' ? 'w-3 h-3 -mr-0.5' : 'w-4 h-4 -mr-1'}
            rounded-sm
            opacity-60 hover:opacity-100
            transition-opacity duration-fast
            focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-current
          `}
          aria-label={`Remove ${children}`}
        >
          <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" className="w-full h-full">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </span>
  );
};

// =============================================================================
// TAG GROUP
// =============================================================================

export interface TagGroupProps extends HTMLAttributes<HTMLDivElement> {
  /** Maximum number of tags to show (rest will be hidden in "+N more") */
  max?: number;
  children: ReactNode;
}

export const TagGroup: React.FC<TagGroupProps> = ({
  max,
  children,
  className = '',
  ...props
}) => {
  const childArray = React.Children.toArray(children);
  const visibleTags = max ? childArray.slice(0, max) : childArray;
  const hiddenCount = max ? Math.max(0, childArray.length - max) : 0;

  return (
    <div
      className={`flex flex-wrap items-center gap-2 ${className}`}
      role="group"
      {...props}
    >
      {visibleTags}
      {hiddenCount > 0 && (
        <span className="text-body-sm text-neutral-500">
          +{hiddenCount} more
        </span>
      )}
    </div>
  );
};

export default Tag;
