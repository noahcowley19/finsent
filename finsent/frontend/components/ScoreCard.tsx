import { cn } from '@/lib/utils';

type Status = 'positive' | 'negative' | 'neutral' | 'warning';

interface ScoreCardProps {
  title: string;
  score: number | string;
  maxScore?: number;
  interpretation: string;
  status: Status;
  details?: Array<{ label: string; value: number | string; passed?: boolean }>;
  className?: string;
}

const statusColors: Record<Status, { bg: string; text: string; border: string }> = {
  positive: { bg: 'bg-positive-light', text: 'text-positive-dark', border: 'border-positive' },
  negative: { bg: 'bg-negative-light', text: 'text-negative-dark', border: 'border-negative' },
  neutral: { bg: 'bg-neutral-light', text: 'text-neutral-dark', border: 'border-neutral' },
  warning: { bg: 'bg-warning-light', text: 'text-warning-dark', border: 'border-warning' },
};

export default function ScoreCard({
  title,
  score,
  maxScore,
  interpretation,
  status,
  details,
  className
}: ScoreCardProps) {
  const colors = statusColors[status];
  
  return (
    <div className={cn('card', className)}>
      <h3 className="font-semibold text-primary mb-3">{title}</h3>
      
      <div className={cn('rounded-lg p-4 mb-3', colors.bg)}>
        <div className="text-center">
          <p className={cn('text-4xl font-bold', colors.text)}>
            {score}{maxScore && <span className="text-lg font-normal">/{maxScore}</span>}
          </p>
          <p className={cn('text-sm font-medium mt-1', colors.text)}>
            {interpretation}
          </p>
        </div>
      </div>
      
      {details && details.length > 0 && (
        <div className="space-y-2">
          {details.map((detail, index) => (
            <div key={index} className="flex justify-between items-center text-sm">
              <span className="text-secondary">{detail.label}</span>
              <span className={cn(
                'font-medium',
                detail.passed === true && 'text-positive',
                detail.passed === false && 'text-negative',
                detail.passed === undefined && 'text-primary'
              )}>
                {detail.value}
                {detail.passed !== undefined && (
                  <span className="ml-1">{detail.passed ? '✓' : '✗'}</span>
                )}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function ScoreBar({ 
  label, 
  value, 
  max, 
  status 
}: { 
  label: string; 
  value: number; 
  max: number;
  status: Status;
}) {
  const percentage = Math.min((value / max) * 100, 100);
  const colors = statusColors[status];
  
  return (
    <div className="mb-3">
      <div className="flex justify-between text-sm mb-1">
        <span className="text-secondary">{label}</span>
        <span className="font-medium text-primary">{value}/{max}</span>
      </div>
      <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
        <div 
          className={cn('h-full rounded-full transition-all', colors.bg.replace('-light', ''))}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
