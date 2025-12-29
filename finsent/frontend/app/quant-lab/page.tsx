'use client';

// =============================================================================
// QUANT LAB PAGE - Strategy Builder & Chaos Lab
// =============================================================================
// Full access to V2 quant tools without paywall for verification
// =============================================================================

import React from 'react';
import { StrategyBuilder } from '@/components/quant-lab/StrategyBuilder';
import { ChaosLab } from '@/components/quant-lab/ChaosLab';

export default function QuantLabPage() {
  return (
    <div className="min-h-screen bg-cream-50 pt-24 pb-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-3xl font-bold text-obsidian-900">Quant Lab</h1>
          <p className="text-obsidian-500 mt-1">
            Vectorized Backtesting & Chaos Theory Analysis
          </p>
        </div>

        {/* Strategy Engine */}
        <div className="bg-white/50 backdrop-blur-md border border-cream-200 rounded-2xl p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-obsidian-900 mb-4">Strategy Builder</h2>
          <StrategyBuilder />
        </div>

        {/* Entropy Engine */}
        <div className="bg-white/50 backdrop-blur-md border border-cream-200 rounded-2xl p-6 shadow-sm">
          <h2 className="text-lg font-semibold text-obsidian-900 mb-4">Chaos Lab</h2>
          <ChaosLab />
        </div>
      </div>
    </div>
  );
}
