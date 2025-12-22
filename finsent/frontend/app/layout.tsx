import type { Metadata, Viewport } from "next";
import { Inter, Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";

// =============================================================================
// FONT CONFIGURATION
// =============================================================================

/**
 * Inter - Body font
 * Used for: Body text, form inputs, data display
 */
const inter = Inter({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-inter",
  preload: true,
});

/**
 * Plus Jakarta Sans - Heading font
 * Used for: Headings, buttons, navigation, UI labels
 */
const plusJakartaSans = Plus_Jakarta_Sans({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-plus-jakarta",
  preload: true,
});

// Note: Fraunces is loaded via CSS @import in globals.css since
// next/font/google doesn't support variable optical sizing well.
// It's used sparingly for display text on landing/marketing pages.

// =============================================================================
// METADATA
// =============================================================================

export const metadata: Metadata = {
  title: {
    default: "Caveray - Financial Intelligence Platform",
    template: "%s | Caveray",
  },
  description:
    "Analyze stocks smarter with AI-powered sentiment analysis, financial metrics, insider trading data, and quantitative insights. Make informed investment decisions with Caveray.",
  keywords: [
    "stock analysis",
    "financial intelligence",
    "sentiment analysis",
    "stock market",
    "investment tools",
    "financial metrics",
    "insider trading",
    "portfolio tracker",
    "quantitative analysis",
  ],
  authors: [{ name: "Caveray" }],
  creator: "Caveray",
  publisher: "Caveray",
  
  // Open Graph
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://caveray.com",
    siteName: "Caveray",
    title: "Caveray - Financial Intelligence Platform",
    description:
      "Analyze stocks smarter with AI-powered tools. Sentiment analysis, financial metrics, insider trading data, and quantitative insights.",
    images: [
      {
        url: "/og-image.png",
        width: 1200,
        height: 630,
        alt: "Caveray - Financial Intelligence Platform",
      },
    ],
  },
  
  // Twitter
  twitter: {
    card: "summary_large_image",
    title: "Caveray - Financial Intelligence Platform",
    description:
      "Analyze stocks smarter with AI-powered tools. Sentiment analysis, financial metrics, insider trading data, and quantitative insights.",
    images: ["/og-image.png"],
  },
  
  // Icons
  icons: {
    icon: [
      { url: "/favicon.ico", sizes: "any" },
      { url: "/icon.svg", type: "image/svg+xml" },
    ],
    apple: [
      { url: "/apple-touch-icon.png", sizes: "180x180" },
    ],
  },
  
  // Manifest
  manifest: "/site.webmanifest",
  
  // Robots
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      "max-video-preview": -1,
      "max-image-preview": "large",
      "max-snippet": -1,
    },
  },
};

// =============================================================================
// VIEWPORT
// =============================================================================

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#FAF7F2" },
    { media: "(prefers-color-scheme: dark)", color: "#131D4F" },
  ],
};

// =============================================================================
// ROOT LAYOUT
// =============================================================================

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${plusJakartaSans.variable}`}
      suppressHydrationWarning
    >
      <head>
        {/* Preconnect to Google Fonts for Fraunces */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body
        className={`
          font-body
          antialiased
          bg-cream-50
          text-navy-900
          min-h-screen
          selection:bg-navy-100
          selection:text-navy-900
        `}
      >
        {/* Skip to main content link for accessibility */}
        <a
          href="#main-content"
          className="
            sr-only
            focus:not-sr-only
            focus:absolute
            focus:top-4
            focus:left-4
            focus:z-max
            focus:px-4
            focus:py-2
            focus:bg-navy-900
            focus:text-white
            focus:rounded-sm
            focus:outline-none
          "
        >
          Skip to main content
        </a>
        
        {/* Main content wrapper */}
        <div id="main-content" className="relative flex flex-col min-h-screen">
          {children}
        </div>
        
        {/* Toast container (for future toast notifications) */}
        <div
          id="toast-container"
          className="fixed top-4 right-4 z-toast flex flex-col gap-3 pointer-events-none"
          aria-live="polite"
          aria-atomic="true"
        />
        
        {/* Modal container (for future modals) */}
        <div
          id="modal-container"
          className="fixed inset-0 z-modal pointer-events-none"
          aria-hidden="true"
        />
      </body>
    </html>
  );
}
