import { cn } from '@/lib/utils';

type Status = 'positive' | 'negative' | 'neutral' | 'warning';

interface MetricCardProps {
  label: string;
  value: string | number;
  status?: Status;
  subtitle?: string;
  className?: string;
}

const statusBorderColors: Record<Status, string> = {
  positive: 'border-t-positive',
  negative: 'border-t-negative',
  neutral: 'border-t-neutral',
  warning: 'border-t-warning',
};

const statusTextColors: Record<Status, string> = {
  positive: 'text-positive',
  negative: 'text-negative',
  neutral: 'text-neutral',
  warning: 'text-warning',
};

export default function MetricCard({ 
  label, 
  value, 
  status = 'neutral',
  subtitle,
  className 
}: MetricCardProps) {
  return (
    <div className={cn(
      'metric-card',
      statusBorderColors[status],
      className
    )}>
      <p className="text-xs font-medium text-secondary uppercase tracking-wide mb-1">
        {label}
      </p>
      <p className={cn('text-2xl font-bold', statusTextColors[status])}>
        {value}
      </p>
      {subtitle && (
        <p className="text-xs text-secondary mt-1">{subtitle}</p>
      )}
    </div>
  );
}

export function MetricRow({ 
  label, 
  value,
  status
}: { 
  label: string; 
  value: string | number;
  status?: Status;
}) {
  return (
    <div className="flex justify-between items-center py-2 border-b border-border last:border-0">
      <span className="text-secondary text-sm">{label}</span>
      <span className={cn(
        'font-semibold',
        status ? statusTextColors[status] : 'text-primary'
      )}>
        {value}
      </span>
    </div>
  );
}

export function MetricGrid({ children, cols = 4 }: { children: React.ReactNode; cols?: 2 | 3 | 4 }) {
  const colClasses = {
    2: 'grid-cols-1 sm:grid-cols-2',
    3: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3',
    4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4',
  };
  
  return (
    <div className={cn('grid gap-4', colClasses[cols])}>
      {children}
    </div>
  );
}
