'use client';

import { useMarketMovers } from '@/lib';

export default function TestApiPage() {
  const { data, loading, error } = useMarketMovers();

  if (loading) return <div>Loading market data...</div>;
  if (error) return <div>Error: {error.message}</div>;
  if (!data) return <div>No data available</div>;

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-4">API Test - Market Movers</h1>
      
      <h2 className="text-xl font-semibold mt-4 mb-2">Top Gainers</h2>
      <ul>
        {data.gainers.slice(0, 5).map(stock => (
          <li key={stock.ticker}>
            {stock.ticker}: {stock.price_display} ({stock.change_display})
          </li>
        ))}
      </ul>
      
      <h2 className="text-xl font-semibold mt-4 mb-2">Top Losers</h2>
      <ul>
        {data.losers.slice(0, 5).map(stock => (
          <li key={stock.ticker}>
            {stock.ticker}: {stock.price_display} ({stock.change_display})
          </li>
        ))}
      </ul>
    </div>
  );
}
