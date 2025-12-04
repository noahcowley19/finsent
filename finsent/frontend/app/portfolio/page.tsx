'use client';

import { useState } from 'react';
import {
  Card,
  CardHeader,
  LoadingOverlay,
  ErrorMessage,
  MetricCard,
  MetricGrid,
  DataTable,
  DoughnutChart,
  BarChart,
} from '@/components';
import { analyzePortfolio } from '@/lib/api';
import type { PortfolioResponse, PortfolioPosition, StockAnalysis } from '@/lib/types';
import { formatCurrency, formatPercent } from '@/lib/utils';
import LoadingSpinner from '@/components/LoadingSpinner';

export default function PortfolioPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<PortfolioResponse | null>(null);
  const [positions, setPositions] = useState<PortfolioPosition[]>([
    { ticker: '', shares: 0, total_cost_basis: 0 }
  ]);
  const [riskFreeRate, setRiskFreeRate] = useState(0.02);
  const [marketReturn, setMarketReturn] = useState(0.1);

  const handleAddPosition = () => {
    setPositions([...positions, { ticker: '', shares: 0, total_cost_basis: 0 }]);
  };

  const handleRemovePosition = (index: number) => {
    if (positions.length > 1) {
      setPositions(positions.filter((_, i) => i !== index));
    }
  };

  const handlePositionChange = (index: number, field: keyof PortfolioPosition, value: string | number) => {
    const newPositions = [...positions];
    if (field === 'ticker') {
      newPositions[index][field] = (value as string).toUpperCase();
    } else {
      newPositions[index][field] = Number(value) || 0;
    }
    setPositions(newPositions);
  };

  const handleAnalyze = async () => {
    const validPositions = positions.filter(p => p.ticker && p.shares > 0);
    
    if (validPositions.length === 0) {
      setError('Please add at least one valid position');
      return;
    }

    setLoading(true);
    setError(null);
    setData(null);
    
    try {
      const result = await analyzePortfolio(validPositions, riskFreeRate, marketReturn);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze portfolio');
    } finally {
      setLoading(false);
    }
  };

  const getReturnStatus = (value: number): 'positive' | 'negative' | 'neutral' => {
    if (value > 0) return 'positive';
    if (value < 0) return 'negative';
    return 'neutral';
  };

  const stockColumns = [
    { key: 'ticker', header: 'Ticker', render: (row: StockAnalysis) => (
      <span className="font-semibold">{row.ticker}</span>
    )},
    { key: 'name', header: 'Name', render: (row: StockAnalysis) => (
      <span className="text-secondary text-sm">{row.name}</span>
    )},
    { key: 'shares', header: 'Shares', align: 'right' as const, render: (row: StockAnalysis) => 
      row.shares?.toLocaleString() ?? '-'
    },
    { key: 'current_price', header: 'Price', align: 'right' as const, render: (row: StockAnalysis) => 
      formatCurrency(row.current_price)
    },
    { key: 'current_value', header: 'Value', align: 'right' as const, render: (row: StockAnalysis) => 
      row.current_value ? formatCurrency(row.current_value) : '-'
    },
    { key: 'gain_loss', header: 'Gain/Loss', align: 'right' as const, render: (row: StockAnalysis) => {
      if (row.gain_loss === undefined || row.gain_loss_percent === undefined) return '-';
      const status = getReturnStatus(row.gain_loss);
      return (
        <span className={status === 'positive' ? 'text-positive' : status === 'negative' ? 'text-negative' : 'text-neutral'}>
          {formatCurrency(row.gain_loss)} ({formatPercent(row.gain_loss_percent, true)})
        </span>
      );
    }},
    { key: 'beta', header: 'Beta', align: 'right' as const, render: (row: StockAnalysis) => 
      row.beta.toFixed(2)
    },
  ];

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {loading && <LoadingOverlay message="Analyzing portfolio..." />}
      
      <h1 className="text-3xl font-bold text-primary mb-6">Portfolio Analysis</h1>

      <Card className="mb-6">
        <CardHeader 
          title="Enter Your Positions" 
          subtitle="Add your stock positions to analyze your portfolio"
        />
        
        <div className="space-y-3 mb-4">
          {positions.map((pos, index) => (
            <div key={index} className="flex gap-3 items-center">
              <input
                type="text"
                placeholder="Ticker"
                value={pos.ticker}
                onChange={(e) => handlePositionChange(index, 'ticker', e.target.value)}
                className="input-field w-24 uppercase"
              />
              <input
                type="number"
                placeholder="Shares"
                value={pos.shares || ''}
                onChange={(e) => handlePositionChange(index, 'shares', e.target.value)}
                className="input-field w-28"
                min="0"
              />
              <input
                type="number"
                placeholder="Total Cost Basis"
                value={pos.total_cost_basis || ''}
                onChange={(e) => handlePositionChange(index, 'total_cost_basis', e.target.value)}
                className="input-field w-32"
                min="0"
                step="0.01"
              />
              <button
                onClick={() => handleRemovePosition(index)}
                className="btn-danger text-sm px-3"
                disabled={positions.length === 1}
              >
                Remove
              </button>
            </div>
          ))}
        </div>

        <button onClick={handleAddPosition} className="btn-secondary text-sm mb-4">
          + Add Position
        </button>

        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium text-secondary mb-1">
              Risk-Free Rate
            </label>
            <input
              type="number"
              value={riskFreeRate}
              onChange={(e) => setRiskFreeRate(Number(e.target.value))}
              className="input-field"
              step="0.01"
              min="0"
              max="1"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-secondary mb-1">
              Expected Market Return
            </label>
            <input
              type="number"
              value={marketReturn}
              onChange={(e) => setMarketReturn(Number(e.target.value))}
              className="input-field"
              step="0.01"
              min="0"
              max="1"
            />
          </div>
        </div>

        <button
          onClick={handleAnalyze}
          disabled={loading}
          className="btn-primary w-full flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <LoadingSpinner size="sm" className="border-white border-t-transparent" />
              <span>Analyzing...</span>
            </>
          ) : (
            'Analyze Portfolio'
          )}
        </button>
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
          <MetricGrid cols={4}>
            <MetricCard 
              label="Total Value" 
              value={formatCurrency(data.portfolio_metrics.total_value)}
            />
            <MetricCard 
              label="Total Cost" 
              value={formatCurrency(data.portfolio_metrics.total_cost)}
            />
            <MetricCard 
              label="Total Gain/Loss" 
              value={formatCurrency(data.portfolio_metrics.total_gain_loss)}
              status={getReturnStatus(data.portfolio_metrics.total_gain_loss)}
            />
            <MetricCard 
              label="Return" 
              value={formatPercent(data.portfolio_metrics.total_gain_loss_percent, true)}
              status={getReturnStatus(data.portfolio_metrics.total_gain_loss_percent)}
            />
          </MetricGrid>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card>
              <CardHeader title="Sector Allocation" />
              <DoughnutChart
                labels={data.allocation.sector.map(s => s.name || 'Unknown')}
                data={data.allocation.sector.map(s => s.percentage)}
              />
            </Card>

            <Card>
              <CardHeader title="Industry Allocation" />
              <DoughnutChart
                labels={data.allocation.industry.slice(0, 8).map(i => i.name || 'Unknown')}
                data={data.allocation.industry.slice(0, 8).map(i => i.percentage)}
              />
            </Card>

            <Card>
              <CardHeader title="Position Weights" />
              <DoughnutChart
                labels={data.allocation.ticker.map(t => t.ticker || 'Unknown')}
                data={data.allocation.ticker.map(t => t.percentage)}
              />
            </Card>
          </div>

          <Card>
            <CardHeader title="Risk Metrics" />
            <MetricGrid cols={4}>
              <MetricCard 
                label="Portfolio Beta" 
                value={data.risk_metrics.portfolio_beta.toFixed(2)}
                status={data.risk_metrics.portfolio_beta > 1 ? 'warning' : 'neutral'}
              />
              <MetricCard 
                label="Diversification Score" 
                value={formatPercent(data.risk_metrics.diversification_score * 100)}
                status={data.risk_metrics.diversification_score > 0.7 ? 'positive' : 
                  data.risk_metrics.diversification_score < 0.4 ? 'negative' : 'neutral'}
              />
              <MetricCard 
                label="Concentration Risk" 
                value={data.risk_metrics.concentration_risk}
                status={data.risk_metrics.concentration_risk.toLowerCase().includes('high') ? 'negative' : 
                  data.risk_metrics.concentration_risk.toLowerCase().includes('low') ? 'positive' : 'neutral'}
              />
              <MetricCard 
                label="Top Holding %" 
                value={formatPercent(data.risk_metrics.max_position_pct)}
              />
            </MetricGrid>
          </Card>

          <Card>
            <CardHeader title="CAPM Analysis" />
            <MetricGrid cols={3}>
              <MetricCard 
                label="Expected Return" 
                value={formatPercent(data.portfolio_capm.expected_return * 100)}
                status={getReturnStatus(data.portfolio_capm.expected_return)}
              />
              <MetricCard 
                label="Risk Premium" 
                value={formatPercent(data.portfolio_capm.risk_premium * 100, true)}
                status={getReturnStatus(data.portfolio_capm.risk_premium)}
              />
              <MetricCard 
                label="Beta" 
                value={data.portfolio_capm.beta.toFixed(2)}
              />
            </MetricGrid>
          </Card>

          <Card>
            <CardHeader 
              title="Stock Analysis" 
              subtitle={`${data.stock_analyses.length} positions`}
            />
            <DataTable
              columns={stockColumns}
              data={data.stock_analyses}
              keyExtractor={(row) => row.ticker}
            />
          </Card>

          <Card>
            <CardHeader title="Position Performance" />
            <BarChart
              labels={data.stock_analyses.map(s => s.ticker)}
              datasets={[{
                label: 'Gain/Loss %',
                data: data.stock_analyses.map(s => s.gain_loss_percent ?? 0),
                color: '#3b82f6',
              }]}
              horizontal
            />
          </Card>
        </div>
      )}
    </div>
  );
}
