import { Link as ChakraLink, Text, Code, ListItem, Heading, UnorderedList } from '@chakra-ui/react'
import { Title, Authors } from 'components/Header'
import { Container } from 'components/Container'
import { DarkModeSwitch } from 'components/DarkModeSwitch'
import { LinksRow } from 'components/LinksRow'
import { Footer } from 'components/Footer'

import { title, abstract, citationId, citationAuthors, citationYear, citationBooktitle, acknowledgements, video_url } from 'data'


const Index = () => (
  <Container>

    {/* Heading */}
    <Title />
    <Authors />

    {/* Links */}
    <LinksRow />

    {/* Video */}
    {/* <Container w="90vw" h="50.6vw" maxW="700px" maxH="393px" mb="3rem">
      <iframe
        width="100%" height="100%"
        src={video_url}
        title="Video"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowFullScreen>
      </iframe>
    </Container> */}

    {/* Main */}
    <Container w="100%" maxW="52rem" alignItems="left" pl="1rem" pr="1rem">

      {/* Abstract */}
      <Heading fontSize="2xl" fontWeight="light" pb="1rem">Abstract</Heading>
      <Text pb="2rem">{abstract}</Text>

      {/* Example */}
      <Heading fontSize="2xl" fontWeight="light" pb="1rem">Overview</Heading>
      <img src={`${process.env.BASE_PATH || ""}/images/splash-figure-v1.png`} />
      <Text align="left" pt="0.5rem" pb="0.5rem" fontSize="small">Fixed Point Diffusion Model (FPDM) is a highly efficient approach to image generation with diffusion models. FPDM integrates an implicit fixed point layer into a denoising diffusion model, converting the sampling process into a sequence of fixed point equations. Our model significantly decreases model size and memory usage while improving performance in settings with limited sampling time or computation. We compare our model, trained at a 256px resolution against the state-of-the-art DiT on four datasets (FFHQ, CelebA-HQ, LSUN-Church, ImageNet) using compute equivalent to 20 DiT sampling steps. FPDM (right) demonstrates enhanced image quality with 87\% fewer parameters and 60\% less memory during training. </Text>

      {/* Another Section */}
      <Heading fontSize="2xl" fontWeight="light" pt="2rem" pb="1rem" id="dataset">Architecture</Heading>
      <img src={`${process.env.BASE_PATH || ""}/images/arch-figure-v1.png`} />
      <Text align="left" pt="0.5rem" pb="0.5rem" fontSize="small">
        Above we show the architecture of FPDM compared with DiT. FPDM keeps the first and last transformer block as pre and post processing layers and replaces the explicit layers in-between with an implicit fixed point layer. Sampling from the full reverse diffusion process involves solving many of these fixed point layers in sequence, which enables the development of new techniques such as timestep smoothing and solution reuse.
      </Text>

      {/* Another Section */}
      <Heading fontSize="2xl" fontWeight="light" pt="2rem" pb="1rem" id="dataset">Examples</Heading>
      <img src={`${process.env.BASE_PATH || ""}/images/supplementary-comparison.png`} />
      <Text align="left" pt="0.5rem" pb="0.5rem" fontSize="small">
        Random (non-cherry-picked) examples of our method compared to DiT.
      </Text>

      {/* Citation */}
      <Heading fontSize="2xl" fontWeight="light" pt="2rem" pb="1rem">Citation</Heading>
      <Code p="0.5rem" borderRadius="5px" overflow="scroll" whiteSpace="nowrap">  {/*  fontFamily="monospace" */}
        @inproceedings&#123; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;{citationId}, <br />
          &nbsp;&nbsp;&nbsp;&nbsp;title=&#123;{title}&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;author=&#123;{citationAuthors}&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;year=&#123;{citationYear}&#125; <br />
          &nbsp;&nbsp;&nbsp;&nbsp;booktitle=&#123;{citationBooktitle}&#125; <br />
        &#125;
      </Code>

      {/* Acknowledgements */}
      <Heading fontSize="2xl" fontWeight="light" pt="2rem" pb="1rem">Acknowledgements</Heading>
      <Text >
        {acknowledgements}
      </Text>
    </Container>

    <DarkModeSwitch />
    <Footer>
      <Text></Text>
    </Footer>
  </Container >
)

export default Index
