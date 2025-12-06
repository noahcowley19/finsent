'use client';

import { useState } from 'react';
import {
  Card,
  CardHeader,
  TickerInput,
  LoadingOverlay,
  ErrorMessage,
  ScoreCard,
  MetricRow,
} from '@/components';
import { analyzeFinancials } from '@/lib/api';
import type { FinancialsResponse, Score } from '@/lib/types';

function getScoreDetails(score: Score): Array<{ label: string; value: string | number; passed?: boolean }> {
  if (!score.components) return [];
  return Object.entries(score.components).map(([key, value]) => {
    let displayValue: string | number;
    let passed: boolean | undefined;
    
    if (typeof value === 'boolean') {
      displayValue = value ? 1 : 0;
      passed = value;
    } else if (typeof value === 'number') {
      displayValue = value;
    } else if (value === null || value === undefined) {
      displayValue = 'N/A';
    } else {
      displayValue = String(value);
    }
    
    return {
      label: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      value: displayValue,
      passed,
    };
  });
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
    <div className="container">
      {loading && <LoadingOverlay message="Analyzing financials..." />}
      
      <div className="hero">
        <h1 style={{ fontSize: '3rem' }}>Financial Analysis</h1>
        <p className="subtitle">Get Piotroski F-Score, Altman Z-Score, and Beneish M-Score analysis</p>
      </div>

      <div className="main-layout" style={{ maxWidth: '1200px' }}>
        <Card className="mb-6">
          <CardHeader 
            title="Analyze Company Financials" 
            subtitle="Enter a ticker to analyze academic scoring models"
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
                <div className="space-y-2">
                  {data.metrics.valuation.map((metric, i) => (
                    <MetricRow key={i} label={metric.name} value={metric.display} status={metric.status} />
                  ))}
                </div>
              </Card>

              <Card>
                <CardHeader title="Profitability Metrics" />
                <div className="space-y-2">
                  {data.metrics.profitability.map((metric, i) => (
                    <MetricRow key={i} label={metric.name} value={metric.display} status={metric.status} />
                  ))}
                </div>
              </Card>

              <Card>
                <CardHeader title="Leverage Metrics" />
                <div className="space-y-2">
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
                  <h4 className="font-semibold text-primary mb-3">Piotroski F-Score (0-9)</h4>
                  <div className="space-y-2 text-secondary">
                    <p><span className="text-positive font-medium">7-9:</span> Strong fundamentals</p>
                    <p><span className="text-warning font-medium">4-6:</span> Average fundamentals</p>
                    <p><span className="text-negative font-medium">0-3:</span> Weak fundamentals</p>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold text-primary mb-3">Altman Z-Score</h4>
                  <div className="space-y-2 text-secondary">
                    <p><span className="text-positive font-medium">&gt;2.99:</span> Safe zone</p>
                    <p><span className="text-warning font-medium">1.81-2.99:</span> Grey zone</p>
                    <p><span className="text-negative font-medium">&lt;1.81:</span> Distress zone</p>
                  </div>
                </div>
                <div>
                  <h4 className="font-semibold text-primary mb-3">Beneish M-Score</h4>
                  <div className="space-y-2 text-secondary">
                    <p><span className="text-positive font-medium">&lt;-2.22:</span> Unlikely manipulator</p>
                    <p><span className="text-warning font-medium">-2.22 to -1.78:</span> Inconclusive</p>
                    <p><span className="text-negative font-medium">&gt;-1.78:</span> Likely manipulator</p>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
