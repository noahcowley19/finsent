'use client';

import React, { HTMLAttributes, ReactNode } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type DividerOrientation = 'horizontal' | 'vertical';
export type DividerVariant = 'solid' | 'dashed' | 'dotted';

export interface DividerProps extends HTMLAttributes<HTMLDivElement> {
  /** Divider orientation */
  orientation?: DividerOrientation;
  /** Line style */
  variant?: DividerVariant;
  /** Text or content in the middle */
  children?: ReactNode;
  /** Spacing around divider */
  spacing?: 'none' | 'sm' | 'md' | 'lg';
}

// =============================================================================
// STYLES
// =============================================================================

const spacingStyles: Record<DividerProps['spacing'] & string, { horizontal: string; vertical: string }> = {
  none: { horizontal: 'my-0', vertical: 'mx-0' },
  sm: { horizontal: 'my-3', vertical: 'mx-3' },
  md: { horizontal: 'my-6', vertical: 'mx-6' },
  lg: { horizontal: 'my-8', vertical: 'mx-8' },
};

const variantStyles: Record<DividerVariant, string> = {
  solid: 'border-solid',
  dashed: 'border-dashed',
  dotted: 'border-dotted',
};

// =============================================================================
// COMPONENT
// =============================================================================

export const Divider: React.FC<DividerProps> = ({
  orientation = 'horizontal',
  variant = 'solid',
  spacing = 'md',
  children,
  className = '',
  ...props
}) => {
  const isHorizontal = orientation === 'horizontal';
  const spaceClass = spacingStyles[spacing][orientation];

  // Simple divider without content
  if (!children) {
    return (
      <div
        role="separator"
        aria-orientation={orientation}
        className={`
          ${isHorizontal ? 'w-full border-t' : 'h-full border-l self-stretch'}
          border-border-light
          ${variantStyles[variant]}
          ${spaceClass}
          ${className}
        `.trim().replace(/\s+/g, ' ')}
        {...props}
      />
    );
  }

  // Divider with content (only for horizontal)
  if (isHorizontal) {
    return (
      <div
        role="separator"
        className={`
          flex items-center
          ${spaceClass}
          ${className}
        `}
        {...props}
      >
        <div
          className={`
            flex-1 border-t border-border-light
            ${variantStyles[variant]}
          `}
        />
        <span className="px-4 text-body-sm text-neutral-500">
          {children}
        </span>
        <div
          className={`
            flex-1 border-t border-border-light
            ${variantStyles[variant]}
          `}
        />
      </div>
    );
  }

  // Vertical with content not supported
  return (
    <div
      role="separator"
      aria-orientation={orientation}
      className={`
        h-full border-l border-border-light
        ${variantStyles[variant]}
        ${spaceClass}
        ${className}
      `}
      {...props}
    />
  );
};

// =============================================================================
// OR DIVIDER (common pattern)
// =============================================================================

export const OrDivider: React.FC<Omit<DividerProps, 'children'>> = (props) => {
  return <Divider {...props}>or</Divider>;
};

export default Divider;
