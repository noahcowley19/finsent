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
    <div className="max-w-[1400px] mx-auto px-5 py-10 relative">
      {/* Hero */}
      <div className="text-center mb-12">
        <h1 className="text-[3.5rem] font-bold text-primary mb-3" style={{ letterSpacing: '-0.03em' }}>
          Caveray
        </h1>
        <p className="text-lg text-secondary font-normal" style={{ letterSpacing: '-0.01em' }}>
          Top-notch financial intelligence tools for smart investing.
        </p>
      </div>

      {/* Features Grid */}
      <div className="max-w-[900px] mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {features.map((feature) => (
            <div
              key={feature.href}
              className={`
                card card-hover flex flex-col relative overflow-hidden group
                ${feature.featured ? 'md:col-span-2 card-featured' : ''}
              `}
            >
              <h2
                className={`text-xl font-bold mb-2 ${feature.featured ? 'text-white' : 'text-primary'}`}
                style={{ letterSpacing: '-0.02em' }}
              >
                {feature.title}
              </h2>
              <p
                className={`text-sm mb-5 leading-relaxed flex-grow ${feature.featured ? 'text-white/80' : 'text-secondary'}`}
              >
                {feature.description}
              </p>
              <Link
                href={feature.href}
                className={`
                  feature-btn
                  ${feature.featured ? 'bg-white !text-primary hover:bg-neutral-light' : ''}
                `}
              >
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
