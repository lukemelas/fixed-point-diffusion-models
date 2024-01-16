import { Heading, Wrap, Box, Container, Link as ChakraLink } from '@chakra-ui/react'
import NextLink from "next/link"

import { title, institutions, authors } from 'data'


export const Title = () => (
  <Heading fontSize="3xl" fontWeight="light" pt="3rem" maxW="42rem" textAlign="center">{title}</Heading>
)


export const Authors = () => (
  <Container>
    <Wrap justify="center" pt="1rem" fontSize="xl" key="authors">
      {
        authors.map((author) =>
          <Box key={author.name} pl="1rem" pr="1rem">
            <NextLink href={author.url} passHref={true}>
              <ChakraLink>{author.name}</ChakraLink>
            </NextLink>
            {/* <sup> {author.institutions.toString()}</sup> */}
          </Box>
        )
      }
    </Wrap>
    {/* <Wrap justify="center" pt="1rem" key="institutions">
      {
        Object.entries(institutions).map(tuple =>
          <Box key={tuple[0]}>
            <sup>{tuple[0]}  </sup>
            {tuple[1]}
          </Box>
        )
      }
    </Wrap> */}
  </Container>
)