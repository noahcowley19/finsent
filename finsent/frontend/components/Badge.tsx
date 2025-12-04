import { cn } from '@/lib/utils';

type Variant = 'positive' | 'negative' | 'neutral' | 'warning' | 'default';
type Size = 'sm' | 'md' | 'lg';

interface BadgeProps {
  children: React.ReactNode;
  variant?: Variant;
  size?: Size;
  className?: string;
}

const variantClasses: Record<Variant, string> = {
  positive: 'bg-positive-light text-positive-dark',
  negative: 'bg-negative-light text-negative-dark',
  neutral: 'bg-neutral-light text-neutral-dark',
  warning: 'bg-warning-light text-warning-dark',
  default: 'bg-slate-100 text-slate-700',
};

const sizeClasses: Record<Size, string> = {
  sm: 'px-2 py-0.5 text-xs',
  md: 'px-2.5 py-1 text-xs',
  lg: 'px-3 py-1.5 text-sm',
};

export default function Badge({ 
  children, 
  variant = 'default', 
  size = 'md',
  className 
}: BadgeProps) {
  return (
    <span className={cn(
      'inline-flex items-center font-medium rounded-full',
      variantClasses[variant],
      sizeClasses[size],
      className
    )}>
      {children}
    </span>
  );
}

export function StatusBadge({ status }: { status: 'positive' | 'negative' | 'neutral' }) {
  const labels = {
    positive: 'Positive',
    negative: 'Negative',
    neutral: 'Neutral',
  };
  
  return <Badge variant={status}>{labels[status]}</Badge>;
}

export function SignalBadge({ signal }: { signal: 'bullish' | 'bearish' | 'neutral' }) {
  const config = {
    bullish: { variant: 'positive' as const, label: 'Bullish' },
    bearish: { variant: 'negative' as const, label: 'Bearish' },
    neutral: { variant: 'neutral' as const, label: 'Neutral' },
  };
  
  const { variant, label } = config[signal];
  return <Badge variant={variant}>{label}</Badge>;
}
