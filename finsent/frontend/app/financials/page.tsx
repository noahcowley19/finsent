'use client';

import { useState } from 'react';
import {
  Card,
  CardHeader,
  CardSection,
  TickerInput,
  LoadingOverlay,
  ErrorMessage,
  ScoreCard,
  MetricCard,
  MetricGrid,
  MetricRow,
} from '@/components';
import { analyzeFinancials } from '@/lib/api';
import type { FinancialsResponse, Score } from '@/lib/types';
import { formatCurrency, formatPercent, formatNumber } from '@/lib/utils';

// Helper to get score components as details array
function getScoreDetails(score: Score): Array<{ label: string; value: string | number; passed?: boolean }> {
  if (!score.components) return [];
  return Object.entries(score.components).map(([key, value]) => ({
    label: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    value: typeof value === 'boolean' ? (value ? 1 : 0) : (value ?? 'N/A'),
    passed: typeof value === 'boolean' ? value : undefined,
  }));
}

export default function FinancialsPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<FinancialsResponse | null>(null);

  const handleAnalyze = async (ticker: string) => {
    setLoading(true);
    setError(null);
    setData(null);
    
    try {
      const result = await analyzeFinancials(ticker);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze financials');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {loading && <LoadingOverlay message="Analyzing financials..." />}
      
      <h1 className="text-3xl font-bold text-primary mb-6">Financial Analysis</h1>

      <Card className="mb-6">
        <CardHeader 
          title="Analyze Company Financials" 
          subtitle="Get Piotroski F-Score, Altman Z-Score, and Beneish M-Score analysis"
        />
        <TickerInput 
          onSubmit={handleAnalyze} 
          loading={loading}
          buttonText="Analyze"
        />
      </Card>

      {error && (
        <ErrorMessage 
          message={error} 
          onRetry={() => setError(null)} 
          className="mb-6"
        />
      )}

      {data && (
        <div className="space-y-6">
          <Card>
            <CardHeader 
              title={data.company.name} 
              subtitle={`${data.company.ticker} • ${data.company.sector} • ${data.company.industry}`}
            />
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <ScoreCard
              title="Piotroski F-Score"
              score={data.scores.piotroski.display}
              maxScore={9}
              interpretation={data.scores.piotroski.interpretation}
              status={data.scores.piotroski.status}
              details={getScoreDetails(data.scores.piotroski)}
            />

            <ScoreCard
              title="Altman Z-Score"
              score={data.scores.altman.display}
              interpretation={data.scores.altman.interpretation}
              status={data.scores.altman.status}
              details={getScoreDetails(data.scores.altman)}
            />

            <ScoreCard
              title="Beneish M-Score"
              score={data.scores.beneish.display}
              interpretation={data.scores.beneish.interpretation}
              status={data.scores.beneish.status}
              details={getScoreDetails(data.scores.beneish)}
            />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card>
              <CardHeader title="Valuation Metrics" />
              <div className="space-y-1">
                {data.metrics.valuation.map((metric, i) => (
                  <MetricRow key={i} label={metric.name} value={metric.display} status={metric.status} />
                ))}
              </div>
            </Card>

            <Card>
              <CardHeader title="Profitability Metrics" />
              <div className="space-y-1">
                {data.metrics.profitability.map((metric, i) => (
                  <MetricRow key={i} label={metric.name} value={metric.display} status={metric.status} />
                ))}
              </div>
            </Card>

            <Card>
              <CardHeader title="Leverage Metrics" />
              <div className="space-y-1">
                {data.metrics.leverage.map((metric, i) => (
                  <MetricRow key={i} label={metric.name} value={metric.display} status={metric.status} />
                ))}
              </div>
            </Card>
          </div>

          <Card>
            <CardHeader title="Score Interpretation Guide" />
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm">
              <div>
                <h4 className="font-semibold text-primary mb-2">Piotroski F-Score (0-9)</h4>
                <ul className="space-y-1 text-secondary">
                  <li><span className="text-positive font-medium">7-9:</span> Strong fundamentals</li>
                  <li><span className="text-warning font-medium">4-6:</span> Average fundamentals</li>
                  <li><span className="text-negative font-medium">0-3:</span> Weak fundamentals</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-primary mb-2">Altman Z-Score</h4>
                <ul className="space-y-1 text-secondary">
                  <li><span className="text-positive font-medium">&gt;2.99:</span> Safe zone</li>
                  <li><span className="text-warning font-medium">1.81-2.99:</span> Grey zone</li>
                  <li><span className="text-negative font-medium">&lt;1.81:</span> Distress zone</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold text-primary mb-2">Beneish M-Score</h4>
                <ul className="space-y-1 text-secondary">
                  <li><span className="text-positive font-medium">&lt;-2.22:</span> Unlikely manipulator</li>
                  <li><span className="text-warning font-medium">-2.22 to -1.78:</span> Inconclusive</li>
                  <li><span className="text-negative font-medium">&gt;-1.78:</span> Likely manipulator</li>
                </ul>
              </div>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
