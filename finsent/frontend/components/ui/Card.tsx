'use client';

import React, { forwardRef, HTMLAttributes, ReactNode } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type CardVariant = 'elevated' | 'outlined' | 'ghost';
export type CardPadding = 'none' | 'sm' | 'md' | 'lg';

export interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** Visual style variant */
  variant?: CardVariant;
  /** Enable hover effects */
  hover?: boolean;
  /** Internal padding */
  padding?: CardPadding;
  /** Make entire card clickable */
  as?: 'div' | 'article' | 'section' | 'button' | 'a';
  /** Content */
  children: ReactNode;
}

// =============================================================================
// STYLES
// =============================================================================

const baseStyles = 'rounded-lg transition-all duration-fast ease-out';

const variantStyles: Record<CardVariant, string> = {
  elevated: 'bg-white shadow-md',
  outlined: 'bg-cream-50 border border-border-light',
  ghost: 'bg-transparent',
};

const hoverStyles: Record<CardVariant, string> = {
  elevated: 'hover:-translate-y-1 hover:shadow-lg cursor-pointer',
  outlined: 'hover:border-border-medium hover:bg-cream-100 cursor-pointer',
  ghost: 'hover:bg-cream-50 cursor-pointer',
};

const paddingStyles: Record<CardPadding, string> = {
  none: 'p-0',
  sm: 'p-4',
  md: 'p-6',
  lg: 'p-8',
};

// =============================================================================
// COMPONENT
// =============================================================================

export const Card = forwardRef<HTMLDivElement, CardProps>(
  (
    {
      variant = 'elevated',
      hover = false,
      padding = 'md',
      as: Component = 'div',
      className = '',
      children,
      ...props
    },
    ref
  ) => {
    const ElementType = Component as React.ElementType;

    return (
      <ElementType
        ref={ref}
        className={`
          ${baseStyles}
          ${variantStyles[variant]}
          ${paddingStyles[padding]}
          ${hover ? hoverStyles[variant] : ''}
          ${className}
        `.trim().replace(/\s+/g, ' ')}
        {...props}
      >
        {children}
      </ElementType>
    );
  }
);

Card.displayName = 'Card';

// =============================================================================
// CARD HEADER
// =============================================================================

export interface CardHeaderProps extends HTMLAttributes<HTMLDivElement> {
  /** Title text */
  title?: string;
  /** Subtitle text */
  subtitle?: string;
  /** Action buttons/icons */
  action?: ReactNode;
  children?: ReactNode;
}

export const CardHeader: React.FC<CardHeaderProps> = ({
  title,
  subtitle,
  action,
  children,
  className = '',
  ...props
}) => {
  if (children) {
    return (
      <div className={`mb-4 ${className}`} {...props}>
        {children}
      </div>
    );
  }

  return (
    <div className={`flex items-start justify-between gap-4 mb-4 ${className}`} {...props}>
      <div>
        {title && (
          <h3 className="font-heading text-heading-md text-navy-900">{title}</h3>
        )}
        {subtitle && (
          <p className="mt-1 text-body-sm text-neutral-500">{subtitle}</p>
        )}
      </div>
      {action && <div className="flex-shrink-0">{action}</div>}
    </div>
  );
};

// =============================================================================
// CARD BODY
// =============================================================================

export interface CardBodyProps extends HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
}

export const CardBody: React.FC<CardBodyProps> = ({
  children,
  className = '',
  ...props
}) => {
  return (
    <div className={className} {...props}>
      {children}
    </div>
  );
};

// =============================================================================
// CARD FOOTER
// =============================================================================

export interface CardFooterProps extends HTMLAttributes<HTMLDivElement> {
  /** Align content */
  align?: 'left' | 'center' | 'right' | 'between';
  children: ReactNode;
}

export const CardFooter: React.FC<CardFooterProps> = ({
  align = 'right',
  children,
  className = '',
  ...props
}) => {
  const alignStyles = {
    left: 'justify-start',
    center: 'justify-center',
    right: 'justify-end',
    between: 'justify-between',
  };

  return (
    <div
      className={`flex items-center gap-3 mt-6 pt-4 border-t border-border-light ${alignStyles[align]} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};

// =============================================================================
// METRIC CARD
// =============================================================================

export interface MetricCardProps extends HTMLAttributes<HTMLDivElement> {
  /** Metric label */
  label: string;
  /** Metric value */
  value: string | number;
  /** Change value (e.g., "+12.5%") */
  change?: string;
  /** Whether change is positive */
  changePositive?: boolean;
  /** Icon */
  icon?: ReactNode;
  /** Card variant */
  variant?: CardVariant;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  change,
  changePositive,
  icon,
  variant = 'elevated',
  className = '',
  ...props
}) => {
  return (
    <Card variant={variant} padding="md" className={className} {...props}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-body-sm text-neutral-500 mb-1">{label}</p>
          <p className="font-heading text-display-sm text-navy-900">{value}</p>
          {change && (
            <p
              className={`mt-1 text-body-sm font-medium ${
                changePositive ? 'text-success-500' : 'text-error-500'
              }`}
            >
              {changePositive ? '↑' : '↓'} {change}
            </p>
          )}
        </div>
        {icon && (
          <div className="w-10 h-10 rounded-lg bg-cream-100 flex items-center justify-center text-navy-500">
            {icon}
          </div>
        )}
      </div>
    </Card>
  );
};

// =============================================================================
// FEATURE CARD
// =============================================================================

export interface FeatureCardProps extends HTMLAttributes<HTMLDivElement> {
  /** Icon */
  icon: ReactNode;
  /** Title */
  title: string;
  /** Description */
  description: string;
  /** CTA link text */
  ctaText?: string;
  /** CTA link href */
  ctaHref?: string;
  /** Click handler */
  onCtaClick?: () => void;
}

export const FeatureCard: React.FC<FeatureCardProps> = ({
  icon,
  title,
  description,
  ctaText,
  ctaHref,
  onCtaClick,
  className = '',
  ...props
}) => {
  return (
    <Card variant="elevated" hover padding="lg" className={className} {...props}>
      <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-navy-500 to-navy-700 flex items-center justify-center text-white mb-4">
        {icon}
      </div>
      <h3 className="font-heading text-heading-md text-navy-900 mb-2">{title}</h3>
      <p className="text-body-sm text-neutral-600 mb-4 line-clamp-3">{description}</p>
      {ctaText && (
        <a
          href={ctaHref}
          onClick={onCtaClick}
          className="inline-flex items-center gap-1 text-body-sm font-medium text-navy-500 hover:text-navy-700 transition-colors"
        >
          {ctaText}
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </a>
      )}
    </Card>
  );
};

export default Card;
