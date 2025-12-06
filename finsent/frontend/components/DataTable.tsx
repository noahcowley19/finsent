import { cn } from '@/lib/utils';

interface Column<T> {
  key: keyof T | string;
  header: string;
  render?: (row: T) => React.ReactNode;
  className?: string;
  align?: 'left' | 'center' | 'right';
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  keyExtractor: (row: T, index: number) => string | number;
  onRowClick?: (row: T) => void;
  emptyMessage?: string;
  className?: string;
}

const alignClasses = {
  left: 'text-left',
  center: 'text-center',
  right: 'text-right',
};

export default function DataTable<T>({
  columns,
  data,
  keyExtractor,
  onRowClick,
  emptyMessage = 'No data available',
  className
}: DataTableProps<T>) {
  if (data.length === 0) {
    return (
      <div className="text-center py-8 text-secondary">
        {emptyMessage}
      </div>
    );
  }

  return (
    <div className={cn('overflow-x-auto -mx-4 px-4', className)}>
      <table className="data-table">
        <thead>
          <tr>
            {columns.map((col) => (
              <th
                key={String(col.key)}
                className={cn(alignClasses[col.align || 'left'], col.className)}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, index) => (
            <tr
              key={keyExtractor(row, index)}
              onClick={() => onRowClick?.(row)}
              className={cn(onRowClick && 'cursor-pointer')}
            >
              {columns.map((col) => (
                <td
                  key={String(col.key)}
                  className={cn(alignClasses[col.align || 'left'], col.className)}
                >
                  {col.render 
                    ? col.render(row) 
                    : String((row as Record<string, unknown>)[col.key as string] ?? '-')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function SimpleTable({ 
  rows 
}: { 
  rows: Array<{ label: string; value: React.ReactNode }> 
}) {
  return (
    <div className="divide-y divide-border">
      {rows.map((row, index) => (
        <div key={index} className="flex justify-between py-3">
          <span className="text-secondary text-sm">{row.label}</span>
          <span className="font-medium text-primary text-sm">{row.value}</span>
        </div>
      ))}
    </div>
  );
}
