'use client';

import React, { ReactNode, HTMLAttributes } from 'react';
import Link from 'next/link';

// =============================================================================
// TYPES
// =============================================================================

export interface Breadcrumb {
  label: string;
  href?: string;
}

export interface PageHeaderProps extends HTMLAttributes<HTMLDivElement> {
  /** Page title */
  title: string;
  /** Page description */
  description?: string;
  /** Breadcrumb items */
  breadcrumbs?: Breadcrumb[];
  /** Action buttons/content */
  actions?: ReactNode;
  /** Badge or status indicator */
  badge?: ReactNode;
  /** Background variant */
  variant?: 'default' | 'gradient' | 'transparent';
  /** Alignment */
  align?: 'left' | 'center';
}

// =============================================================================
// BREADCRUMBS COMPONENT
// =============================================================================

const Breadcrumbs: React.FC<{ items: Breadcrumb[] }> = ({ items }) => {
  return (
    <nav aria-label="Breadcrumb" className="mb-4">
      <ol className="flex items-center gap-2 text-body-sm">
        {items.map((item, index) => {
          const isLast = index === items.length - 1;

          return (
            <li key={index} className="flex items-center gap-2">
              {item.href && !isLast ? (
                <Link
                  href={item.href}
                  className="text-neutral-500 hover:text-navy-700 transition-colors"
                >
                  {item.label}
                </Link>
              ) : (
                <span className={isLast ? 'text-navy-900 font-medium' : 'text-neutral-500'}>
                  {item.label}
                </span>
              )}
              {!isLast && (
                <svg
                  className="w-4 h-4 text-neutral-300"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
};

// =============================================================================
// COMPONENT
// =============================================================================

export const PageHeader: React.FC<PageHeaderProps> = ({
  title,
  description,
  breadcrumbs,
  actions,
  badge,
  variant = 'default',
  align = 'left',
  className = '',
  ...props
}) => {
  const variantStyles = {
    default: 'bg-cream-50',
    gradient: 'bg-gradient-to-br from-cream-50 to-cream-100',
    transparent: 'bg-transparent',
  };

  const alignStyles = {
    left: 'text-left',
    center: 'text-center items-center',
  };

  return (
    <div
      className={`
        ${variantStyles[variant]}
        border-b border-border-light
        ${className}
      `}
      {...props}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-12">
        {/* Breadcrumbs */}
        {breadcrumbs && breadcrumbs.length > 0 && (
          <Breadcrumbs items={breadcrumbs} />
        )}

        <div
          className={`
            flex flex-col lg:flex-row lg:items-end lg:justify-between gap-4
            ${alignStyles[align]}
          `}
        >
          {/* Title section */}
          <div className={align === 'center' ? 'flex-1' : ''}>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="font-heading text-display-sm lg:text-display-md text-navy-900">
                {title}
              </h1>
              {badge}
            </div>
            {description && (
              <p className="text-body-md lg:text-body-lg text-neutral-600 max-w-2xl">
                {description}
              </p>
            )}
          </div>

          {/* Actions */}
          {actions && (
            <div className="flex items-center gap-3 flex-shrink-0">
              {actions}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// =============================================================================
// SIMPLE PAGE TITLE (for minimal headers)
// =============================================================================

export interface PageTitleProps extends HTMLAttributes<HTMLDivElement> {
  title: string;
  subtitle?: string;
}

export const PageTitle: React.FC<PageTitleProps> = ({
  title,
  subtitle,
  className = '',
  ...props
}) => {
  return (
    <div className={`mb-6 ${className}`} {...props}>
      <h1 className="font-heading text-heading-xl text-navy-900">{title}</h1>
      {subtitle && (
        <p className="mt-1 text-body-md text-neutral-600">{subtitle}</p>
      )}
    </div>
  );
};

export default PageHeader;
