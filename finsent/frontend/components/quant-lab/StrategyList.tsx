'use client';

// =============================================================================
// STRATEGY LIST COMPONENT
// =============================================================================
// Displays saved trading strategies
//
// Location: frontend/components/quant-lab/StrategyList.tsx
//
// =============================================================================

import React from 'react';

export interface SavedStrategy {
  id: string;
  name: string;
  symbol: string;
  createdAt: Date;
  lastRun?: Date;
  performance?: {
    totalReturn: number;
    winRate: number;
  };
}

export interface StrategyListProps {
  strategies: SavedStrategy[];
  onLoad: (strategyId: string) => void;
  onDelete?: (strategyId: string) => void;
  activeStrategyId?: string;
  isLoading?: boolean;
}

export const StrategyList: React.FC<StrategyListProps> = ({
  strategies,
  onLoad,
  onDelete,
  activeStrategyId,
  isLoading = false,
}) => {
  if (isLoading) {
    return (
      <div className="bg-white rounded-xl border border-border-light p-6">
        <div className="w-32 h-6 rounded bg-cream-100 animate-pulse mb-4" />
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="p-4 rounded-lg bg-cream-50 animate-pulse">
              <div className="w-24 h-4 rounded bg-cream-100 mb-2" />
              <div className="w-16 h-3 rounded bg-cream-100" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-xl border border-border-light p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-heading font-semibold text-heading-sm text-navy-900">
          Saved Strategies
        </h3>
        <span className="text-caption text-neutral-500">
          {strategies.length} strategies
        </span>
      </div>

      {strategies.length === 0 ? (
        <div className="text-center py-8">
          <svg className="w-12 h-12 mx-auto mb-4 text-neutral-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <p className="text-body-sm text-neutral-500">No saved strategies</p>
          <p className="text-caption text-neutral-400 mt-1">
            Build and save a strategy to see it here
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          {strategies.map((strategy) => {
            const isActive = strategy.id === activeStrategyId;
            const isPositive = (strategy.performance?.totalReturn ?? 0) >= 0;

            return (
              <div
                key={strategy.id}
                className={`
                  p-4 rounded-lg cursor-pointer transition-colors
                  ${isActive
                    ? 'bg-navy-50 border border-navy-200'
                    : 'bg-cream-50 hover:bg-cream-100'
                  }
                `}
                onClick={() => onLoad(strategy.id)}
              >
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <p className="text-body-sm font-medium text-navy-900">
                      {strategy.name}
                    </p>
                    <p className="text-caption text-neutral-500">
                      {strategy.symbol} • Created {strategy.createdAt.toLocaleDateString()}
                    </p>
                  </div>

                  {onDelete && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDelete(strategy.id);
                      }}
                      className="p-1 text-neutral-400 hover:text-error-600 transition-colors"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  )}
                </div>

                {strategy.performance && (
                  <div className="flex items-center gap-4 text-caption">
                    <span className={`font-medium ${isPositive ? 'text-success-600' : 'text-error-600'}`}>
                      {isPositive ? '+' : ''}{strategy.performance.totalReturn.toFixed(2)}%
                    </span>
                    <span className="text-neutral-500">
                      Win rate: {strategy.performance.winRate.toFixed(0)}%
                    </span>
                  </div>
                )}

                {strategy.lastRun && (
                  <p className="text-caption text-neutral-400 mt-1">
                    Last run: {strategy.lastRun.toLocaleDateString()}
                  </p>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default StrategyList;
