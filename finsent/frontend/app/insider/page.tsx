'use client';

import { useState } from 'react';
import {
  Card,
  CardHeader,
  CardSection,
  TickerInput,
  LoadingOverlay,
  ErrorMessage,
  MetricCard,
  MetricGrid,
  Badge,
  DataTable,
  BarChart,
  SignalBadge,
} from '@/components';
import { analyzeInsider } from '@/lib/api';
import type { InsiderResponse, InsiderTransaction, Signal, ClusterAlert } from '@/lib/types';
import { formatCurrency, formatLargeNumber, formatPercent } from '@/lib/utils';

export default function InsiderPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<InsiderResponse | null>(null);
  const [months, setMonths] = useState(12);

  const handleAnalyze = async (ticker: string) => {
    setLoading(true);
    setError(null);
    setData(null);
    
    try {
      const result = await analyzeInsider(ticker, months);
      setData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze insider activity');
    } finally {
      setLoading(false);
    }
  };

  const transactionColumns = [
    { key: 'date', header: 'Date' },
    { key: 'insider', header: 'Insider' },
    { key: 'title', header: 'Title' },
    { key: 'type', header: 'Type', render: (row: InsiderTransaction) => (
      <Badge variant={row.type_status}>
        {row.type}
      </Badge>
    )},
    { key: 'shares', header: 'Shares', align: 'right' as const, render: (row: InsiderTransaction) => 
      row.shares_display
    },
    { key: 'value', header: 'Value', align: 'right' as const, render: (row: InsiderTransaction) => 
      row.value_display
    },
  ];

  const signalColumns = [
    { key: 'type', header: 'Signal Type' },
    { key: 'title', header: 'Title' },
    { key: 'description', header: 'Description' },
    { key: 'status', header: 'Status', render: (row: Signal) => (
      <Badge variant={row.status}>
        {row.status}
      </Badge>
    )},
  ];

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {loading && <LoadingOverlay message="Analyzing insider activity..." />}
      
      <h1 className="text-3xl font-bold text-primary mb-6">Insider Trading Analysis</h1>

      <Card className="mb-6">
        <CardHeader 
          title="Analyze Insider Activity" 
          subtitle="Track insider buying and selling patterns"
        />
        <div className="flex gap-4 items-end mb-4">
          <div className="flex-1">
            <label className="block text-sm font-medium text-secondary mb-1">
              Lookback Period
            </label>
            <select
              value={months}
              onChange={(e) => setMonths(Number(e.target.value))}
              className="input-field"
            >
              <option value={3}>3 months</option>
              <option value={6}>6 months</option>
              <option value={12}>12 months</option>
              <option value={24}>24 months</option>
            </select>
          </div>
        </div>
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

          <MetricGrid cols={4}>
            <MetricCard 
              label="Total Transactions" 
              value={data.transactions.length}
              status="neutral"
            />
            <MetricCard 
              label="Buy Value" 
              value={data.summary.buy_value_display}
              status="positive"
            />
            <MetricCard 
              label="Sell Value" 
              value={data.summary.sell_value_display}
              status="negative"
            />
            <MetricCard 
              label="Overall Sentiment" 
              value={data.sentiment.sentiment}
              status={data.sentiment.status}
            />
          </MetricGrid>

          {data.signals.length > 0 && (
            <Card>
              <CardHeader title="Trading Signals" subtitle="Key patterns identified" />
              <DataTable
                columns={signalColumns}
                data={data.signals}
                keyExtractor={(_, i) => i}
              />
            </Card>
          )}

          {data.monthly_data.length > 0 && (
            <Card>
              <CardHeader title="Monthly Activity" />
              <BarChart
                labels={data.monthly_data.map(d => d.label)}
                datasets={[
                  { label: 'Buys', data: data.monthly_data.map(d => d.buys), color: '#10b981' },
                  { label: 'Sells', data: data.monthly_data.map(d => d.sells), color: '#ef4444' },
                ]}
              />
            </Card>
          )}

          {data.cluster_alerts.length > 0 && (
            <Card>
              <CardHeader title="Cluster Alerts" subtitle="Unusual activity patterns" />
              <div className="space-y-3">
                {data.cluster_alerts.map((alert: ClusterAlert, i) => (
                  <div 
                    key={i} 
                    className={`p-4 rounded-lg border ${
                      alert.status === 'positive' ? 'bg-positive-light border-positive' : 'bg-negative-light border-negative'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div>
                        <p className={`font-semibold ${alert.status === 'positive' ? 'text-positive-dark' : 'text-negative-dark'}`}>
                          {alert.message}
                        </p>
                        <p className="text-sm mt-1 opacity-80">{alert.description}</p>
                      </div>
                      <Badge variant={alert.status}>{alert.total_value_display}</Badge>
                    </div>
                    {alert.insiders.length > 0 && (
                      <div className="mt-3 text-sm">
                        <p className="font-medium mb-1">Insiders involved:</p>
                        <ul className="space-y-1">
                          {alert.insiders.map((insider, j) => (
                            <li key={j}>{insider.name} ({insider.title}) - {insider.value}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </Card>
          )}

          {data.has_transaction_data && (
            <Card>
              <CardHeader 
                title="Recent Transactions" 
                subtitle={`${data.transactions.length} transactions in the last ${months} months`}
              />
              <DataTable
                columns={transactionColumns}
                data={data.transactions}
                keyExtractor={(row, i) => `${row.date}-${row.insider}-${i}`}
                emptyMessage="No insider transactions found"
              />
            </Card>
          )}

          {data.has_institutional_data && data.institutional.holders.length > 0 && (
            <Card>
              <CardHeader title="Top Institutional Holders" />
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {data.institutional.holders.slice(0, 6).map((holder, i) => (
                  <div key={i} className="p-4 bg-slate-50 rounded-lg">
                    <p className="font-semibold text-primary truncate">{holder.name}</p>
                    <p className="text-sm text-secondary">
                      {holder.shares_display} shares ({holder.percent_display})
                    </p>
                    <p className="text-xs text-secondary mt-1">
                      Value: {holder.value_display}
                    </p>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
