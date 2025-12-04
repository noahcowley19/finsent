
import { cn } from '@/lib/utils';

interface ErrorMessageProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  className?: string;
}

export default function ErrorMessage({ 
  title = 'Error', 
  message, 
  onRetry,
  className 
}: ErrorMessageProps) {
  return (
    <div className={cn('bg-negative-light border border-negative rounded-lg p-4', className)}>
      <h3 className="font-semibold text-negative-dark mb-1">{title}</h3>
      <p className="text-negative-dark/80 text-sm">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-3 btn-secondary text-sm"
        >
          Try Again
        </button>
      )}
    </div>
  );
}

export function EmptyState({ 
  title, 
  message,
  icon
}: { 
  title: string; 
  message: string;
  icon?: React.ReactNode;
}) {
  return (
    <div className="text-center py-12">
      {icon && <div className="text-4xl mb-4">{icon}</div>}
      <h3 className="text-lg font-semibold text-primary mb-2">{title}</h3>
      <p className="text-secondary">{message}</p>
    </div>
  );
}
