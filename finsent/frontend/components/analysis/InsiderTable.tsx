'use client';

// =============================================================================
// INSIDER TABLE COMPONENT
// =============================================================================
// Table showing insider transactions
//
// Location: frontend/components/analysis/InsiderTable.tsx
//
// =============================================================================

import React, { useState } from 'react';

export interface InsiderTransaction {
  id: string;
  date: Date;
  insider: string;
  title: string;
  type: 'buy' | 'sell' | 'option';
  shares: number;
  price: number;
  value: number;
}

export interface InsiderTableProps {
  transactions: InsiderTransaction[];
  isLoading?: boolean;
}

type FilterType = 'all' | 'buy' | 'sell';

function formatCurrency(value: number): string {
  if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `$${(value / 1e3).toFixed(2)}K`;
  return `$${value.toLocaleString()}`;
}

export const InsiderTable: React.FC<InsiderTableProps> = ({
  transactions,
  isLoading = false,
}) => {
  const [filter, setFilter] = useState<FilterType>('all');

  const filteredTransactions = transactions.filter((t) => {
    if (filter === 'all') return true;
    return t.type === filter;
  });

  if (isLoading) {
    return (
      <div className="bg-white rounded-xl border border-border-light overflow-hidden">
        <div className="p-6 border-b border-border-light">
          <div className="flex justify-between items-center">
            <div className="w-48 h-6 rounded bg-cream-100 animate-pulse" />
            <div className="flex gap-2">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="w-16 h-8 rounded bg-cream-100 animate-pulse" />
              ))}
            </div>
          </div>
        </div>
        <div className="animate-pulse">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="flex items-center gap-4 p-4 border-b border-border-light">
              <div className="w-24 h-4 rounded bg-cream-100" />
              <div className="w-32 h-4 rounded bg-cream-100" />
              <div className="w-24 h-4 rounded bg-cream-100" />
              <div className="flex-1" />
              <div className="w-20 h-4 rounded bg-cream-100" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-border-light overflow-hidden">
      {/* Header */}
      <div className="p-6 border-b border-border-light">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <h3 className="font-heading font-semibold text-heading-sm text-navy-900">
            Recent Transactions
          </h3>

          {/* Filter */}
          <div className="flex gap-1 bg-cream-50 rounded-lg p-1">
            {(['all', 'buy', 'sell'] as FilterType[]).map((type) => (
              <button
                key={type}
                onClick={() => setFilter(type)}
                className={`
                  px-3 py-1.5 rounded-md text-body-sm font-medium capitalize
                  transition-colors duration-fast
                  ${filter === type
                    ? 'bg-white text-navy-900 shadow-sm'
                    : 'text-neutral-600 hover:text-navy-700'
                  }
                `}
              >
                {type}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-cream-50">
              <th className="text-left py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider">
                Date
              </th>
              <th className="text-left py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider">
                Insider
              </th>
              <th className="text-left py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider">
                Type
              </th>
              <th className="text-right py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider">
                Shares
              </th>
              <th className="text-right py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider">
                Price
              </th>
              <th className="text-right py-3 px-4 text-caption font-semibold text-neutral-600 uppercase tracking-wider">
                Value
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border-light">
            {filteredTransactions.length === 0 ? (
              <tr>
                <td colSpan={6} className="py-12 text-center text-body-sm text-neutral-500">
                  No transactions found
                </td>
              </tr>
            ) : (
              filteredTransactions.map((transaction) => (
                <tr key={transaction.id} className="hover:bg-cream-50 transition-colors">
                  <td className="py-4 px-4 text-body-sm text-neutral-600">
                    {transaction.date.toLocaleDateString()}
                  </td>
                  <td className="py-4 px-4">
                    <p className="text-body-sm font-medium text-navy-900">
                      {transaction.insider}
                    </p>
                    <p className="text-caption text-neutral-500">
                      {transaction.title}
                    </p>
                  </td>
                  <td className="py-4 px-4">
                    <span
                      className={`
                        inline-flex items-center px-2 py-1 rounded text-caption font-medium uppercase
                        ${transaction.type === 'buy'
                          ? 'bg-success-100 text-success-700'
                          : transaction.type === 'sell'
                          ? 'bg-error-100 text-error-700'
                          : 'bg-neutral-100 text-neutral-700'
                        }
                      `}
                    >
                      {transaction.type}
                    </span>
                  </td>
                  <td className="py-4 px-4 text-body-sm text-navy-900 text-right tabular-nums">
                    {transaction.shares.toLocaleString()}
                  </td>
                  <td className="py-4 px-4 text-body-sm text-navy-900 text-right tabular-nums">
                    ${transaction.price.toFixed(2)}
                  </td>
                  <td className="py-4 px-4 text-right">
                    <span
                      className={`text-body-sm font-medium tabular-nums ${
                        transaction.type === 'buy' ? 'text-success-600' : 'text-error-600'
                      }`}
                    >
                      {formatCurrency(transaction.value)}
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Footer */}
      {filteredTransactions.length > 0 && (
        <div className="p-4 border-t border-border-light bg-cream-50">
          <p className="text-caption text-neutral-500 text-center">
            Showing {filteredTransactions.length} of {transactions.length} transactions
          </p>
        </div>
      )}
    </div>
  );
};

export default InsiderTable;
