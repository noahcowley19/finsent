import Link from 'next/link';

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
    <div className="container max-w-[1400px] mx-auto px-5 py-10">
      <div className="text-center mb-12">
        <h1 className="text-[3.5rem] font-bold text-primary mb-3 tracking-tight">
          Caveray
        </h1>
        <p className="text-lg text-secondary">
          Top-notch financial intelligence tools for smart investing.
        </p>
      </div>

      <div className="max-w-[900px] mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {features.map((feature) => (
            <Link
              key={feature.href}
              href={feature.href}
              className={`
                card p-8 transition-all duration-300 hover:shadow-custom-hover hover:-translate-y-1
                group relative overflow-hidden
                ${feature.featured ? 'md:col-span-2 bg-gradient-to-br from-primary to-[#334155] text-white' : ''}
              `}
            >
              <h2 className={`text-xl font-bold mb-2 ${feature.featured ? 'text-white' : 'text-primary'}`}>
                {feature.title}
              </h2>
              <p className={`text-sm mb-5 leading-relaxed ${feature.featured ? 'text-white/80' : 'text-secondary'}`}>
                {feature.description}
              </p>
              <div className={`
                inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold rounded-lg
                transition-all duration-200
                ${feature.featured 
                  ? 'bg-white text-primary hover:bg-neutral-light' 
                  : 'bg-primary text-white hover:bg-[#0f172a]'
                }
              `}>
                Launch {feature.title}
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
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
