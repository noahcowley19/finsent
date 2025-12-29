'use client';

import React, { forwardRef, HTMLAttributes, ReactNode } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type CardVariant = 'default' | 'glass' | 'outline' | 'ghost';
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
// STYLES - Modern SaaS Aesthetic
// =============================================================================

const baseStyles = 'rounded-xl transition-all duration-200 ease-out';

const variantStyles: Record<CardVariant, string> = {
  default: 'bg-white border border-ink-200/50 shadow-sm',
  glass: 'bg-white/80 backdrop-blur-xl border border-white/20 shadow-glass',
  outline: 'bg-transparent border border-ink-200',
  ghost: 'bg-transparent',
};

const hoverStyles: Record<CardVariant, string> = {
  default: 'hover:-translate-y-0.5 hover:shadow-lg hover:border-ink-300/50 cursor-pointer',
  glass: 'hover:-translate-y-0.5 hover:shadow-glass-lg hover:border-white/30 cursor-pointer',
  outline: 'hover:bg-ink-50 hover:border-ink-300 cursor-pointer',
  ghost: 'hover:bg-ink-50 cursor-pointer',
};

const paddingStyles: Record<CardPadding, string> = {
  none: 'p-0',
  sm: 'p-4',
  md: 'p-6',
  lg: 'p-8',
};

// =============================================================================
// CARD COMPONENT
// =============================================================================

export const Card = forwardRef<HTMLDivElement, CardProps>(
  (
    {
      variant = 'default',
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
          <h3 className="font-semibold text-heading-md text-ink-900 tracking-tight">{title}</h3>
        )}
        {subtitle && (
          <p className="mt-1 text-body-sm text-ink-500">{subtitle}</p>
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
      className={`flex items-center gap-3 mt-6 pt-4 border-t border-ink-100 ${alignStyles[align]} ${className}`}
      {...props}
    >
      {children}
    </div>
  );
};

// =============================================================================
// METRIC CARD - For dashboard metrics
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
  variant = 'default',
  className = '',
  ...props
}) => {
  return (
    <Card variant={variant} padding="md" hover className={`group ${className}`} {...props}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-gray-500 uppercase tracking-wide font-medium mb-3">{label}</p>
          <p className="font-bold text-3xl text-ink-black-900 tracking-tight">{value}</p>
          {change && (
            <p
              className={`mt-2 text-sm font-medium flex items-center gap-1 ${changePositive ? 'text-success-600' : 'text-terracotta-500'
                }`}
            >
              <span className="text-xs">{changePositive ? '↑' : '↓'}</span>
              {change}
            </p>
          )}
        </div>
        {icon && (
          <div className="text-navy-900 opacity-60">
            {icon}
          </div>
        )}
      </div>
    </Card>
  );
};

// =============================================================================
// FEATURE CARD - For landing page features
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
    <Card variant="default" hover padding="lg" className={`group ${className}`} {...props}>
      <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-ink-100 to-ink-200 flex items-center justify-center text-ink-600 mb-4 group-hover:from-accent/10 group-hover:to-accent/20 group-hover:text-accent transition-all">
        {icon}
      </div>
      <h3 className="font-semibold text-heading-md text-ink-900 tracking-tight mb-2">{title}</h3>
      <p className="text-body-sm text-ink-500 mb-4 line-clamp-3">{description}</p>
      {ctaText && (
        <a
          href={ctaHref}
          onClick={onCtaClick}
          className="inline-flex items-center gap-1 text-body-sm font-medium text-accent hover:text-accent-600 transition-colors"
        >
          {ctaText}
          <svg className="w-4 h-4 transition-transform group-hover:translate-x-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </a>
      )}
    </Card>
  );
};

export default Card;
