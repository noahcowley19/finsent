import Link from 'next/link';

const ArrowIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    fill="none"
    viewBox="0 0 24 24"
    stroke="currentColor"
    strokeWidth="2"
    className="w-3.5 h-3.5 transition-transform duration-200 group-hover:translate-x-1"
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
  </svg>
);

const features = [
  {
    title: 'Stock Search',
    description: 'A comprehensive stock search with charts, metrics, ratings, financial health indicators, and real-time news.',
    href: '/search',
    featured: true,
  },
  {
    title: 'Sentiment Analyzer',
    description: 'Real-time market sentiment using transformer models and social data.',
    href: '/sentiment',
  },
  {
    title: 'Financial Analyzer',
    description: 'Academic scoring models: Piotroski F-Score, Altman Z-Score, and Beneish M-Score.',
    href: '/financials',
  },
  {
    title: 'Insider Trading',
    description: 'Track insider buying/selling activity and institutional ownership changes.',
    href: '/insider',
  },
  {
    title: 'My Portfolio',
    description: 'A portfolio tracker with cool features, including return prediction.',
    href: '/portfolio',
  },
];

export default function Home() {
  return (
    <div className="container">
      {/* Hero Section */}
      <div className="hero">
        <h1>Caveray</h1>
        <p className="subtitle">Top-notch financial intelligence tools for smart investing.</p>
      </div>

      {/* Main Layout */}
      <div className="main-layout">
        <div className="features-grid">
          {features.map((feature) => (
            <div
              key={feature.href}
              className={`feature-card group ${feature.featured ? 'featured' : ''}`}
            >
              <h2>{feature.title}</h2>
              <p>{feature.description}</p>
              <Link href={feature.href} className="feature-btn">
                Launch {feature.title}
                <ArrowIcon />
              </Link>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
