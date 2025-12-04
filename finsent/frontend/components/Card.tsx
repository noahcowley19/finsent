
import { cn } from '@/lib/utils';

interface CardProps {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
  onClick?: () => void;
}

export default function Card({ children, className, hover = false, onClick }: CardProps) {
  return (
    <div
      className={cn(
        hover ? 'card-hover cursor-pointer' : 'card',
        className
      )}
      onClick={onClick}
    >
      {children}
    </div>
  );
}

export function CardHeader({ 
  title, 
  subtitle,
  action
}: { 
  title: string; 
  subtitle?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between mb-4">
      <div>
        <h2 className="text-xl font-bold text-primary">{title}</h2>
        {subtitle && <p className="text-secondary text-sm mt-1">{subtitle}</p>}
      </div>
      {action}
    </div>
  );
}

export function CardSection({ 
  title, 
  children,
  className
}: { 
  title?: string; 
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('mt-4', className)}>
      {title && <h3 className="font-semibold text-primary mb-3">{title}</h3>}
      {children}
    </div>
  );
}
