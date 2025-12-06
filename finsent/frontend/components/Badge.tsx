import { cn } from '@/lib/utils';
import { ReactNode } from 'react';

type BadgeVariant = 'positive' | 'negative' | 'neutral' | 'warning';

interface BadgeProps {
  children: ReactNode;
  variant?: BadgeVariant;
  className?: string;
}

const variantClasses: Record<BadgeVariant, string> = {
  positive: 'bg-positive-light text-positive-dark',
  negative: 'bg-negative-light text-negative-dark',
  neutral: 'bg-neutral-light text-neutral-dark',
  warning: 'bg-warning-light text-warning-dark',
};

export default function Badge({ children, variant = 'neutral', className }: BadgeProps) {
  return (
    <span className={cn('badge', variantClasses[variant], className)}>
      {children}
    </span>
  );
}

interface StatusBadgeProps {
  status: 'positive' | 'negative' | 'neutral' | 'warning';
  label?: string;
  className?: string;
}

export function StatusBadge({ status, label, className }: StatusBadgeProps) {
  const labels: Record<string, string> = {
    positive: label || 'Good',
    negative: label || 'Poor',
    neutral: label || 'Neutral',
    warning: label || 'Warning',
  };

  return (
    <Badge variant={status} className={className}>
      {labels[status]}
    </Badge>
  );
}

interface SignalBadgeProps {
  signal: string;
  status: 'positive' | 'negative' | 'neutral';
  className?: string;
}

export function SignalBadge({ signal, status, className }: SignalBadgeProps) {
  return (
    <Badge variant={status} className={className}>
      {signal}
    </Badge>
  );
}
