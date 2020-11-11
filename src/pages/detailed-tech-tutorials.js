import React from 'react';

import Layout from "gatsby-theme-blog-starter/src/components/layout/layout"
import Seo from "gatsby-theme-blog-starter/src/components/seo/Seo"
import AboutLayout from '../components/layouts/AboutLayout';

const About = () => {
  return(
  <Layout>
    <Seo 
      title="Detailed Technology Tutorials on latest Technologies"
      description="Detailed Tutorials on Technology. System Design, Blockchain, Big Data In Depth Tutorials"
      keywords={["Detailed Technology Tutorials", "System Design", "Blockchain", "Big Data", "Spark"]}
      slug="detailed-tech-tutorials"  />
    
    <AboutLayout />
  </Layout>
)}

export default About