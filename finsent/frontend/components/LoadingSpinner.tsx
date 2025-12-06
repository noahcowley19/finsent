import { cn } from '@/lib/utils';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const sizeClasses = {
  sm: 'w-4 h-4 border-2',
  md: 'w-8 h-8 border-3',
  lg: 'w-12 h-12 border-4',
};

export default function LoadingSpinner({ size = 'md', className }: LoadingSpinnerProps) {
  return (
    <div 
      className={cn(
        'rounded-full border-border border-t-primary animate-spin',
        sizeClasses[size],
        className
      )}
    />
  );
}

interface LoadingOverlayProps {
  message?: string;
}

export function LoadingOverlay({ message = 'Loading...' }: LoadingOverlayProps) {
  return (
    <div className="loading-overlay">
      <div className="spinner" />
      <p className="loading-text">{message}</p>
    </div>
  );
}

interface LoadingCardProps {
  message?: string;
  className?: string;
}

export function LoadingCard({ message = 'Loading...', className }: LoadingCardProps) {
  return (
    <div className={cn('card flex flex-col items-center justify-center py-12', className)}>
      <div className="spinner mb-4" />
      <p className="text-secondary text-sm">{message}</p>
    </div>
  );
}
