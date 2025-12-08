'use client';

import { ReactNode } from 'react';

type BadgeVariant = 'default' | 'primary' | 'success' | 'warning' | 'danger' | 'info';
type BadgeSize = 'sm' | 'md' | 'lg';

interface BadgeProps {
  children: ReactNode;
  variant?: BadgeVariant;
  size?: BadgeSize;
  className?: string;
}

const variantStyles: Record<BadgeVariant, { bg: string; color: string; border: string }> = {
  default: { bg: 'var(--bg-secondary)', color: 'var(--text-secondary)', border: 'var(--border)' },
  primary: { bg: 'var(--accent-subtle)', color: 'var(--accent)', border: 'rgba(0, 212, 170, 0.3)' },
  success: { bg: 'var(--positive-light)', color: 'var(--positive)', border: 'rgba(0, 229, 160, 0.3)' },
  warning: { bg: 'var(--warning-light)', color: 'var(--warning)', border: 'rgba(251, 191, 36, 0.3)' },
  danger: { bg: 'var(--negative-light)', color: 'var(--negative)', border: 'rgba(255, 107, 107, 0.3)' },
  info: { bg: 'rgba(59, 130, 246, 0.15)', color: '#3b82f6', border: 'rgba(59, 130, 246, 0.3)' },
};

const sizeStyles: Record<BadgeSize, { padding: string; fontSize: string }> = {
  sm: { padding: '2px 8px', fontSize: '11px' },
  md: { padding: '4px 12px', fontSize: '12px' },
  lg: { padding: '6px 16px', fontSize: '13px' },
};

export default function Badge({ children, variant = 'default', size = 'md', className = '' }: BadgeProps) {
  const styles = variantStyles[variant];
  const sizes = sizeStyles[size];

  return (
    <span
      className={className}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '6px',
        padding: sizes.padding,
        fontSize: sizes.fontSize,
        fontWeight: 600,
        borderRadius: '8px',
        background: styles.bg,
        color: styles.color,
        border: `1px solid ${styles.border}`,
      }}
    >
      {children}
    </span>
  );
}

interface StatusBadgeProps {
  status: 'positive' | 'negative' | 'neutral' | 'warning';
  label: string;
  showDot?: boolean;
  size?: BadgeSize;
}

export function StatusBadge({ status, label, showDot = true, size = 'md' }: StatusBadgeProps) {
  const statusVariant: Record<string, BadgeVariant> = {
    positive: 'success',
    negative: 'danger',
    neutral: 'default',
    warning: 'warning',
  };

  const dotColors: Record<string, string> = {
    positive: 'var(--positive)',
    negative: 'var(--negative)',
    neutral: 'var(--text-muted)',
    warning: 'var(--warning)',
  };

  return (
    <Badge variant={statusVariant[status]} size={size}>
      {showDot && (
        <span
          style={{
            width: '6px',
            height: '6px',
            borderRadius: '50%',
            background: dotColors[status],
            boxShadow: `0 0 6px ${dotColors[status]}`,
          }}
        />
      )}
      {label}
    </Badge>
  );
}

interface SignalBadgeProps {
  type: string;
  status: 'positive' | 'negative' | 'neutral' | 'warning';
  title: string;
  description: string;
}

export function SignalBadge({ type, status, title, description }: SignalBadgeProps) {
  const statusColors: Record<string, { bg: string; color: string; border: string }> = {
    positive: { bg: 'var(--positive-light)', color: 'var(--positive)', border: 'rgba(0, 229, 160, 0.3)' },
    negative: { bg: 'var(--negative-light)', color: 'var(--negative)', border: 'rgba(255, 107, 107, 0.3)' },
    warning: { bg: 'var(--warning-light)', color: 'var(--warning)', border: 'rgba(251, 191, 36, 0.3)' },
    neutral: { bg: 'var(--neutral-light)', color: 'var(--text-secondary)', border: 'var(--border)' },
  };
  
  const colors = statusColors[status] || statusColors.neutral;

  return (
    <div
      style={{
        background: colors.bg,
        border: `1px solid ${colors.border}`,
        borderRadius: '12px',
        padding: '12px 16px',
        flex: '1 1 280px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
        <div
          style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: colors.color,
            boxShadow: `0 0 8px ${colors.color}`,
          }}
        />
        <span style={{ fontSize: '13px', fontWeight: 600, color: colors.color }}>{title}</span>
      </div>
      <p style={{ fontSize: '12px', color: 'var(--text-secondary)', margin: 0, lineHeight: 1.5 }}>
        {description}
      </p>
    </div>
  );
}
