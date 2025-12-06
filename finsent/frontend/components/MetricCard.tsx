import { cn } from '@/lib/utils';
import { ReactNode } from 'react';

type MetricStatus = 'positive' | 'negative' | 'neutral' | 'warning';

interface MetricCardProps {
  label: string;
  value: string | number;
  subtitle?: string;
  status?: MetricStatus;
  className?: string;
}

export default function MetricCard({ 
  label, 
  value, 
  subtitle,
  status,
  className 
}: MetricCardProps) {
  return (
    <div className={cn('metric-item', status, className)}>
      <div className="metric-name">{label}</div>
      <div className="metric-value">{value}</div>
      {subtitle && (
        <div className="text-xs text-secondary mt-1">{subtitle}</div>
      )}
    </div>
  );
}

interface MetricRowProps {
  label: string;
  value: string | number | ReactNode;
  status?: MetricStatus;
  className?: string;
}

export function MetricRow({ label, value, status, className }: MetricRowProps) {
  const statusColors: Record<MetricStatus, string> = {
    positive: 'text-positive',
    negative: 'text-negative',
    neutral: 'text-neutral-dark',
    warning: 'text-warning-dark',
  };

  return (
    <div className={cn(
      'flex justify-between items-center py-3 px-4 bg-background rounded-lg',
      className
    )}>
      <span className="text-sm text-secondary font-medium">{label}</span>
      <span className={cn(
        'text-sm font-semibold',
        status ? statusColors[status] : 'text-primary'
      )}>
        {value}
      </span>
    </div>
  );
}

interface MetricGridProps {
  children: ReactNode;
  cols?: 2 | 3 | 4;
  className?: string;
}

export function MetricGrid({ children, cols = 4, className }: MetricGridProps) {
  const colClasses = {
    2: 'grid-cols-1 sm:grid-cols-2',
    3: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
  };

  return (
    <div className={cn('grid gap-4', colClasses[cols], className)}>
      {children}
    </div>
  );
}
